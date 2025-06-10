"""Utility functions for better integration with LangChain and LangGraph."""

import json
import logging
import re
from typing import Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)


def parse_malformed_args_string(args_str: str) -> dict:
    """Parse args when LLM formats them incorrectly.

    Example: 'key="value", key2="value2"' or similar.

    This is a more robust parser for various malformed argument formats that LLMs might
    produce when they don't properly format tool calls.
    """
    if args_str is None or not args_str:
        return {}

    # First try to parse as JSON if it's a valid JSON string already
    try:
        if args_str.strip().startswith("{") and args_str.strip().endswith("}"):
            return json.loads(args_str)
    except json.JSONDecodeError:
        pass  # Fall through to other parsing methods

    result = {}

    # Try to match key="value" or key='value' patterns
    # This handles formats like name="John", age="30"
    kv_matches = re.findall(r'([a-zA-Z0-9_]+)\s*=\s*["\']([^"\']*)["\']', args_str)
    if kv_matches:
        for key, value in kv_matches:
            result[key] = value

    # Try to match key=value patterns without quotes (must process even if quoted
    # matches found). This handles formats like name=John, age=30 or mixed cases like
    # name="John", age=30
    kv_matches_no_quotes = re.findall(r'([a-zA-Z0-9_]+)\s*=\s*([^,\s"\']+)', args_str)
    if kv_matches_no_quotes:
        for key, value in kv_matches_no_quotes:
            # Skip keys already processed in the quoted version to avoid duplicates
            if key not in result:
                result[key] = value

    # If we found any key-value pairs, return them
    if result:
        return result

    # If all else fails, just return the string as a value
    return {"value": args_str}


def meta_agent_factory(
    llm: BaseChatModel,
    tools: Optional[list[StructuredTool]] = None,
    system_prompt_text: str = "",
    output_schema: Optional[Union[type[BaseModel], dict]] = None,
    disable_streaming: bool = False,
    additional_tags: Optional[list[str]] = None,
    method: str = "json_mode",
) -> RunnableSequence:
    """Create a Meta-specific agent with structured output support.

    Args:
        llm: Base language model to use
        tools: Optional list of tools for the agent to use
        system_prompt_text: Optional system prompt to override the default
        output_schema: Optional Pydantic schema or dict for structured output
        disable_streaming: Whether to disable streaming
        additional_tags: Optional list of additional tags
        method: Method to use for structured output

    Returns:
        A runnable chain that can be used for structured output

    Example:
        >>> from langchain_core.language_models import BaseChatModel
        >>> from langchain_core.tools import StructuredTool
        >>> from pydantic import BaseModel
        >>> class MySchema(BaseModel):
        ...     foo: str
        >>> llm = ... # some BaseChatModel
        >>> tools = [StructuredTool(...)]
        >>> agent = meta_agent_factory(llm, tools=tools, output_schema=MySchema)
        >>> result = agent.invoke({"messages": [...]})
    """
    # Always disable streaming for structured output - this is crucial for Meta LLMs
    if output_schema is not None or disable_streaming:
        # Force disable streaming at the LLM level for more reliable outputs
        if hasattr(llm, "streaming") and llm.streaming:
            logger.debug("Disabling streaming in LLM for structured output reliability")
            llm = llm.bind(streaming=False)

    # Set low temperature for structured output - critical for Meta models
    if hasattr(llm, "temperature") and (
        llm.temperature is None or llm.temperature > 0.2
    ):
        logger.debug("Setting temperature=0.1 for more reliable structured output")
        llm = llm.bind(temperature=0.1)

    # Use structured output if schema is provided
    if output_schema is not None:
        logger.debug(
            f"Attempting to use structured output with schema: {output_schema}"
        )

        try:
            # Ensure schema is of the correct type
            if not (
                isinstance(output_schema, type) and issubclass(output_schema, BaseModel)
            ) and not isinstance(output_schema, dict):
                raise ValueError(
                    "output_schema must be a Pydantic model class or a dict for "
                    "with_structured_output."
                )

            # Prepare schema for Meta's response_format
            if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                schema_name = output_schema.__name__
                schema_dict = output_schema.model_json_schema()
            else:
                schema_name = output_schema.get("name", "OutputSchema")
                schema_dict = output_schema

            # Apply Meta-specific response_format structure
            llm = llm.bind(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema_dict,
                    },
                }
            )

            # Also use the LangChain with_structured_output for backward compatibility
            try:
                bound_llm_wrapper = llm.with_structured_output(
                    output_schema, include_raw=False
                )
                bound_llm = RunnablePassthrough() | bound_llm_wrapper
            except Exception as struct_e:
                logger.warning(
                    f"Error using with_structured_output: {struct_e}. Using direct "
                    "response_format instead."
                )
                bound_llm = llm

        except Exception as e:
            logger.error(f"Error setting up structured output: {e}", exc_info=True)
            logger.info(
                "Falling back to default LLM behavior without structured output."
            )
            # Fall back to regular LLM if structured output setup fails
            bound_llm = llm

    elif tools:
        logger.debug(
            "Binding provided tools: "
            f"{[tool.name for tool in tools if hasattr(tool, 'name')]}"
        )
        try:
            # Make sure to disable streaming for tool calling with Meta LLMs
            if disable_streaming and hasattr(llm, "streaming") and llm.streaming:
                llm = llm.bind(streaming=False)

            # Set low temperature for tool calling
            if hasattr(llm, "temperature") and (
                llm.temperature is None or llm.temperature > 0.2
            ):
                llm = llm.bind(temperature=0.1)

            # This will now call the (correctly implemented) bind_tools on the llm.
            bound_llm = llm.bind_tools(tools)
        except Exception as e:
            logger.error(f"Error binding tools: {e}", exc_info=True)
            logger.info("Falling back to default LLM behavior without tools.")
            bound_llm = llm  # Fallback
    else:
        # Default case: no schema, no tools
        bound_llm = llm

    # Add any additional tags
    if additional_tags:
        if isinstance(additional_tags, list):
            bound_llm = bound_llm.bind(tags=additional_tags)

    # Create a basic prompt
    try:
        # Create the prompt template safely, avoiding issues with parsing
        # curly braces in the system prompt (like {\"next\": \"value\"})
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt_text),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
    except Exception as prompt_error:
        logger.error(
            f"Error creating prompt template: {prompt_error}. Using a direct "
            "SystemMessage instead."
        )
        # Fallback approach if template creation fails
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_text),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    # Return the full chain
    def ensure_list_output(output):
        # If output is a dict with "messages", return as is
        if isinstance(output, dict) and "messages" in output:
            return output
        # If output is a single BaseMessage, wrap in a list
        if isinstance(output, BaseMessage):
            return {"messages": [output]}
        # If output is a list of BaseMessages, wrap in dict
        if isinstance(output, list) and all(isinstance(m, BaseMessage) for m in output):
            return {"messages": output}
        # Otherwise, return as is
        return output

    return (prompt | bound_llm) | RunnableLambda(ensure_list_output)


def extract_json_response(content: Any) -> Any:
    """Extract JSON from various response formats.

    Handles:
    - Direct JSON objects
    - JSON in code blocks with backticks
    - JSON-like patterns in text

    Args:
        content: The response content to parse

    Returns:
        Parsed JSON dict or original content if parsing failed
    """
    if not isinstance(content, str):
        return content

    # Try direct JSON parsing
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from code blocks
    if "```" in content:
        # Try JSON code blocks
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

    # Try to find JSON objects in text
    match = re.search(r"({[\s\S]*?})", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # If we get here, return the original content
    return content
