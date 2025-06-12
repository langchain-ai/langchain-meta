import json
import logging
import re
import uuid
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)  # Added Callable and Optional here

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool  # Added for _lc_tool_to_llama_tool_param
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)  # Ensure import
from llama_api_client.types import MessageParam  # Use MessageParam
from llama_api_client.types.chat import (
    completion_create_params,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Serialize a LangChain message to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "type": message.__class__.__name__,
        "content": message.content
        if isinstance(message.content, str)
        else str(message.content),
        "additional_kwargs": {},
    }
    if message.additional_kwargs:
        for k, v in message.additional_kwargs.items():
            if k not in ["__pydantic_serializer__", "__pydantic_validator__"]:
                try:
                    json.dumps(v)
                    result["additional_kwargs"][k] = v
                except (TypeError, OverflowError):
                    result["additional_kwargs"][k] = str(v)
    if isinstance(message, AIMessage):
        if message.tool_calls:
            serialized_tool_calls = []
            for tc_item in message.tool_calls:
                tc_dict_curr: dict[str, Any] = {}
                item_id = (
                    str(tc_item.get("id", uuid.uuid4()))
                    if isinstance(tc_item, dict)
                    else str(getattr(tc_item, "id", uuid.uuid4()))
                )
                item_name = (
                    str(tc_item.get("name", "unknown_tool"))
                    if isinstance(tc_item, dict)
                    else str(getattr(tc_item, "name", "unknown_tool"))
                )
                item_args = (
                    tc_item.get("args", {})
                    if isinstance(tc_item, dict)
                    else getattr(tc_item, "args", {})
                )

                tc_dict_curr["id"] = item_id
                tc_dict_curr["name"] = item_name

                if isinstance(item_args, dict):
                    try:
                        tc_dict_curr["args"] = json.dumps(item_args)
                    except (TypeError, OverflowError):
                        tc_dict_curr["args"] = str(item_args)
                elif item_args is not None:
                    tc_dict_curr["args"] = str(item_args)
                else:
                    tc_dict_curr["args"] = "{}"

                serialized_tool_calls.append(tc_dict_curr)
            result["tool_calls"] = serialized_tool_calls
        if message.additional_kwargs.get("function_call"):
            function_call = message.additional_kwargs["function_call"]
            if isinstance(function_call, dict) and "arguments" in function_call:
                try:
                    if not isinstance(function_call["arguments"], str):
                        function_call["arguments"] = json.dumps(
                            function_call["arguments"]
                        )
                    json.loads(function_call["arguments"])
                    result["function_call"] = function_call
                except (TypeError, OverflowError, json.JSONDecodeError):
                    func_call_copy = function_call.copy()
                    func_call_copy["arguments"] = str(function_call["arguments"])
                    result["function_call"] = func_call_copy
            else:
                result["function_call"] = str(function_call)
    elif isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id
        if not isinstance(message.content, str):
            try:
                if isinstance(message.content, (dict, list)):
                    result["content"] = json.dumps(message.content)
            except (TypeError, OverflowError):
                result["content"] = str(message.content)
    return result


def _lc_message_to_llama_message_param(
    message: BaseMessage,
    available_tools_with_schema: Optional[list[completion_create_params.Tool]] = None,
) -> MessageParam:
    """Converts a LangChain BaseMessage to a Llama API MessageParam."""
    role: str
    content: Union[str, dict[str, Any]]  # noqa: F842
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    if isinstance(message, HumanMessage):
        role = "user"
        content_payload = message.content
    elif isinstance(message, AIMessage):
        role = "assistant"
        content_payload = message.content if message.content is not None else ""
        if message.tool_calls and len(message.tool_calls) > 0:
            processed_tool_calls = []
            for tc in message.tool_calls:
                args_val = tc.get("args")
                # Ensure args_dict is a dictionary
                if isinstance(args_val, str):
                    try:
                        args_dict = json.loads(args_val)
                    except Exception:
                        logger.warning(
                            f"Failed to parse tool call args string: {args_val}. Representing as {{'value': args_val}}"
                        )
                        args_dict = {"value": args_val}
                elif isinstance(args_val, dict):
                    args_dict = args_val
                elif args_val is None:  # Explicitly handle None
                    args_dict = {}
                else:  # Handle other non-dict, non-string, non-None types
                    args_dict = {"value": str(args_val)}

                # Coerce arguments if schema is available
                if available_tools_with_schema:
                    tool_name = tc.get("name")
                    tool_def_found: Optional[completion_create_params.Tool] = None
                    for tool_definition in available_tools_with_schema:
                        # tool_definition is completion_create_params.Tool, a TypedDict
                        current_fn_def = tool_definition.get("function")
                        if (
                            tool_definition.get("type") == "function"
                            and current_fn_def
                            and isinstance(current_fn_def, dict)
                            and current_fn_def.get("name") == tool_name
                        ):
                            tool_def_found = tool_definition
                            break
                    if tool_def_found:
                        param_properties: Optional[dict[str, Any]] = None
                        fn_def_from_found = tool_def_found.get("function")
                        if fn_def_from_found and isinstance(fn_def_from_found, dict):
                            params_schema = fn_def_from_found.get("parameters")
                            if params_schema and isinstance(params_schema, dict):
                                # The 'parameters' field holds the JSON schema object,
                                # which should have a 'properties' key if it defines an object with properties.
                                properties_val = params_schema.get("properties")
                                if isinstance(properties_val, dict):
                                    param_properties = properties_val
                                else:
                                    param_properties = {}  # No properties defined for this tool, or not an object schema
                            else:
                                param_properties = {}  # No parameters schema for this tool
                        else:
                            param_properties = {}  # No function definition in found tool, should not happen if outer check passed

                        if (
                            param_properties
                        ):  # Ensure param_properties is a dict and not None
                            coerced_args_dict = {}
                            for arg_name, arg_value in args_dict.items():
                                prop_schema = param_properties.get(arg_name)
                                if (
                                    prop_schema
                                    and isinstance(prop_schema, dict)
                                    and isinstance(arg_value, str)
                                ):
                                    expected_type = prop_schema.get("type")
                                    try:
                                        if expected_type == "integer":
                                            coerced_args_dict[arg_name] = int(arg_value)
                                        elif (
                                            expected_type == "number"
                                        ):  # JSON schema "number" can be float or int
                                            coerced_args_dict[arg_name] = float(
                                                arg_value
                                            )  # type: ignore
                                        elif expected_type == "boolean":
                                            if arg_value.lower() == "true":
                                                coerced_args_dict[arg_name] = True
                                            elif arg_value.lower() == "false":
                                                coerced_args_dict[arg_name] = False
                                            else:
                                                logger.warning(
                                                    f"Cannot coerce string '{arg_value}' to boolean for arg '{arg_name}'. Keeping as string."
                                                )
                                                coerced_args_dict[arg_name] = arg_value  # type: ignore
                                        else:
                                            coerced_args_dict[arg_name] = arg_value  # type: ignore
                                    except ValueError:
                                        logger.warning(
                                            f"Failed to coerce string arg '{arg_name}' value '{arg_value}' to type '{expected_type}'. Keeping as string."
                                        )
                                        coerced_args_dict[arg_name] = arg_value  # type: ignore
                                else:
                                    coerced_args_dict[arg_name] = arg_value
                            args_dict = coerced_args_dict

                processed_tool_calls.append(
                    {
                        "id": tc.get("id") or str(uuid.uuid4()),  # Ensure ID
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(args_dict),
                        },
                    }
                )
            # Assign processed tool calls to the outer scope variable intended for the message dict
            tool_calls = processed_tool_calls

    elif isinstance(message, SystemMessage):
        role = "system"
        content_payload = message.content
    elif isinstance(message, ToolMessage):
        role = "tool"
        if isinstance(message.content, (list, dict)):
            content_payload = json.dumps(message.content)
        else:
            content_payload = str(message.content)
        tool_call_id = message.tool_call_id
    else:
        raise ValueError(f"Got unknown message type: {type(message)}")

    msg_dict: dict[str, Any] = {
        "role": role,
        "content": content_payload,
    }

    if tool_calls:
        msg_dict["tool_calls"] = tool_calls
    if tool_call_id:
        msg_dict["tool_call_id"] = tool_call_id

    if role == "assistant" and tool_calls:
        msg_dict["content"] = ""

    return cast(MessageParam, msg_dict)


def _get_json_type_for_annotation(type_name: str) -> str:
    """Helper to convert Python type annotations to JSON schema types."""
    mapping = {
        "str": "string",
        "Text": "string",
        "string": "string",
        "int": "number",
        "float": "number",
        "complex": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "tuple": "array",
        "array": "array",
        "List": "array",
        "Tuple": "array",
        "dict": "object",
        "Dict": "object",
        "mapping": "object",
        "Mapping": "object",
    }
    return mapping.get(type_name, "string")


def _convert_dict_tool(lc_tool: Any) -> dict:
    if (
        isinstance(lc_tool, dict)
        and "function" in lc_tool
        and isinstance(lc_tool["function"], dict)
    ):
        # Ensure strict: True is added if not present for consistency
        if "strict" not in lc_tool["function"]:
            lc_tool["function"]["strict"] = True
            logger.debug("Added strict: True to directly provided dict tool function.")
        return lc_tool
    raise ValueError("Not a dict tool suitable for Llama API direct use")


def _convert_pydantic_class_tool(
    lc_tool: type[BaseModel],
) -> dict:
    name = getattr(lc_tool, "__name__", "UnnamedTool")
    description = getattr(lc_tool, "__doc__", "") or ""
    pydantic_schema = {}
    if hasattr(lc_tool, "model_json_schema") and callable(
        getattr(lc_tool, "model_json_schema")
    ):
        pydantic_schema = lc_tool.model_json_schema()
    elif hasattr(lc_tool, "schema") and callable(getattr(lc_tool, "schema")):
        pydantic_schema = lc_tool.schema()

    # Unwrap the Pydantic schema for Llama API
    llama_parameters: dict[str, Any] = {}
    if isinstance(pydantic_schema, dict):
        if "properties" in pydantic_schema:
            llama_parameters["properties"] = pydantic_schema["properties"]
            # Special handling for include_domains/exclude_domains if generated with anyOf by Pydantic
            for field_name in ["include_domains", "exclude_domains"]:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_array, has_null, array_schema_details = False, False, None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "array"
                            and isinstance(sub_schema.get("items"), dict)
                            and sub_schema["items"].get("type") == "string"
                        ):
                            has_array, array_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_array and has_null and array_schema_details:
                        new_field_def = {
                            "type": "array",
                            "items": array_schema_details["items"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
            # Special handling for Optional[bool] fields like include_images
            for field_name, field_def in list(
                llama_parameters.get("properties", {}).items()
            ):  # Iterate over a copy
                if isinstance(field_def, dict) and "anyOf" in field_def:
                    has_boolean, has_null = False, False
                    for sub_schema in field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "boolean"
                        ):
                            has_boolean = True
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_boolean and has_null:
                        new_simplified_def = {"type": "boolean"}
                        if "description" in field_def:
                            new_simplified_def["description"] = field_def["description"]
                        if "title" in field_def:
                            new_simplified_def["title"] = field_def["title"]
                        # If there was a default, it would be handled by Pydantic model or API if not sent
                        llama_parameters["properties"][field_name] = new_simplified_def
                        logger.debug(
                            f"Simplified Optional[bool] schema for field '{field_name}'."
                        )

            # Special handling for Optional[Literal] fields like search_depth, time_range, topic
            optional_literal_fields = ["search_depth", "time_range", "topic"]
            for field_name in optional_literal_fields:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_enum_str, has_null = False, False
                    enum_schema_details = None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "string"
                            and "enum" in sub_schema
                        ):
                            has_enum_str, enum_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_enum_str and has_null and enum_schema_details:
                        new_field_def = {
                            "type": "string",
                            "enum": enum_schema_details["enum"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
                        logger.debug(
                            f"Simplified Optional[Literal] schema for field '{field_name}'."
                        )

        if "required" in pydantic_schema:
            llama_parameters["required"] = pydantic_schema["required"]
    else:
        logger.warning(
            f"Schema for {name} was not a dict: {type(pydantic_schema)}. Using empty parameters."
        )

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": llama_parameters,
            "strict": True,
        },
    }


def _convert_structured_tool(
    lc_tool: BaseTool,
) -> dict:  # Changed Any to BaseTool for clarity
    name = str(getattr(lc_tool, "name", "unnamed_tool"))
    description = str(getattr(lc_tool, "description", ""))
    pydantic_schema = {}
    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema is not None:
        args_schema = lc_tool.args_schema
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            pydantic_schema = args_schema.model_json_schema()
        elif isinstance(
            args_schema, dict
        ):  # If args_schema is already a dict, use it directly
            pydantic_schema = args_schema
        else:
            logger.warning(
                f"args_schema for {name} is not a Pydantic class or dict. Using empty schema."
            )

    # Unwrap the Pydantic schema for Llama API
    llama_parameters: dict[str, Any] = {}
    if isinstance(pydantic_schema, dict):
        if "properties" in pydantic_schema:
            llama_parameters["properties"] = pydantic_schema["properties"]
            # Special handling for include_domains/exclude_domains if generated with anyOf by Pydantic
            for field_name in ["include_domains", "exclude_domains"]:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_array, has_null, array_schema_details = False, False, None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "array"
                            and isinstance(sub_schema.get("items"), dict)
                            and sub_schema["items"].get("type") == "string"
                        ):
                            has_array, array_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_array and has_null and array_schema_details:
                        new_field_def = {
                            "type": "array",
                            "items": array_schema_details["items"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
            # Special handling for Optional[bool] fields like include_images (copied from _convert_pydantic_class_tool)
            for field_name, field_def in list(
                llama_parameters.get("properties", {}).items()
            ):  # Iterate over a copy
                if isinstance(field_def, dict) and "anyOf" in field_def:
                    has_boolean, has_null = False, False
                    for sub_schema in field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "boolean"
                        ):
                            has_boolean = True
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_boolean and has_null:
                        new_simplified_def = {"type": "boolean"}
                        if "description" in field_def:
                            new_simplified_def["description"] = field_def["description"]
                        if "title" in field_def:
                            new_simplified_def["title"] = field_def["title"]
                        llama_parameters["properties"][field_name] = new_simplified_def
                        logger.debug(
                            f"Simplified Optional[bool] schema for field '{field_name}' in structured tool."
                        )

            # Special handling for Optional[Literal] fields like search_depth, time_range, topic
            optional_literal_fields = ["search_depth", "time_range", "topic"]
            for field_name in optional_literal_fields:
                if (
                    field_name in llama_parameters["properties"]
                    and "anyOf" in llama_parameters["properties"][field_name]
                ):
                    original_field_def = llama_parameters["properties"][field_name]
                    has_enum_str, has_null = False, False
                    enum_schema_details = None
                    for sub_schema in original_field_def.get("anyOf", []):
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "string"
                            and "enum" in sub_schema
                        ):
                            has_enum_str, enum_schema_details = True, sub_schema
                        if (
                            isinstance(sub_schema, dict)
                            and sub_schema.get("type") == "null"
                        ):
                            has_null = True
                    if has_enum_str and has_null and enum_schema_details:
                        new_field_def = {
                            "type": "string",
                            "enum": enum_schema_details["enum"],
                        }
                        if "description" in original_field_def:
                            new_field_def["description"] = original_field_def[
                                "description"
                            ]
                        if "title" in original_field_def:
                            new_field_def["title"] = original_field_def["title"]
                        llama_parameters["properties"][field_name] = new_field_def
                        logger.debug(
                            f"Simplified Optional[Literal] schema for field '{field_name}'."
                        )

        if "required" in pydantic_schema:
            llama_parameters["required"] = pydantic_schema["required"]
    else:
        logger.warning(
            f"Schema for {name} was not a dict or was empty. Using empty parameters."
        )

    function_def = {
        "name": name,
        "description": description,
        "parameters": llama_parameters,
        "strict": True,
    }
    # The Llama API examples show parameters: {} when no params, so ensure it's at least an empty dict.
    if not llama_parameters.get("properties") and not llama_parameters.get("required"):
        # Ensure function_def exists and is a dict before assigning to its keys
        if (
            function_def is None
            or not function_def.get("name")
            or not function_def.get("description")
        ):
            logger.warning(
                f"Function definition is incomplete or None for tool {name}. Creating minimal fallback."
            )
            function_def = {
                "name": name,
                "description": description,
                "parameters": {},
                "strict": True,
            }
        else:
            function_def[
                "parameters"
            ] = {}  # Ensure parameters is {} if no props/required, not just containing additionalProperties:false
    elif "properties" not in llama_parameters:  # if only required is present
        llama_parameters[
            "properties"
        ] = {}  # Llama might expect properties key even if empty if other keys like required are present
        if function_def is None:
            function_def = {}  # Should not happen
        function_def["parameters"] = llama_parameters

    # Ensure function_def is not None before returning
    if function_def is None:
        # This case implies lc_tool was not a BaseTool or did not have a valid schema,
        # and llama_parameters remained empty. We should create a minimal valid function_def.
        logger.warning(
            f"Function definition was unexpectedly None for tool {name}. Creating minimal fallback."
        )
        function_def = {
            "name": name,
            "description": description,
            "parameters": {},
            "strict": True,
        }

    return {"type": "function", "function": function_def}


def _convert_parse_method_tool(lc_tool: Any) -> dict:
    # This case is less common for Llama direct tools; often covered by StructuredTool
    if hasattr(lc_tool, "parse") and callable(getattr(lc_tool, "parse")):
        name = getattr(
            lc_tool, "name", getattr(lc_tool, "__class__", type(lc_tool)).__name__
        )
        description = (
            getattr(lc_tool, "__doc__", "") or f"Tool that parses input for {name}"
        )
        schema = getattr(lc_tool, "schema", None)
        parameters = (
            schema()
            if callable(schema)
            else (
                schema
                if isinstance(schema, dict)
                else {"type": "object", "properties": {}}
            )
        )
        if isinstance(parameters, dict) and parameters.get("type") == "object":
            parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": True,
            },
        }
    raise ValueError("Not a parse-method tool or schema extraction failed")


def _convert_route_schema_tool(lc_tool: Any) -> dict:
    # Specific handling for a tool named "RouteSchema" or "route_schema"
    # This seems very application-specific, ensure it's truly generic or handled by specific tool logic.
    if hasattr(lc_tool, "name") and getattr(lc_tool, "name") in [
        "RouteSchema",
        "route_schema",
    ]:
        # This enum should ideally be dynamic or part of the tool's definition
        enum_values = [
            "EmailAgent",
            "ScribeAgent",
            "TimeKeeperAgent",
            "GeneralAgent",
            "END",
            "__end__",
        ]
        return {
            "type": "function",
            "function": {
                "name": getattr(lc_tool, "name"),
                "description": "Route to the next agent",
                "parameters": {
                    "type": "object",
                    "properties": {"next": {"type": "string", "enum": enum_values}},
                    "required": ["next"],
                },
                "strict": True,  # Added strict here as well
            },
        }
    raise ValueError("Not a RouteSchema tool by name convention")


def _create_minimal_tool(lc_tool: Any) -> dict:
    name = str(getattr(lc_tool, "name", type(lc_tool).__name__))
    description = str(getattr(lc_tool, "description", ""))
    logger.error(
        f"Could not convert tool {name} ({type(lc_tool)}) to Llama API format. Creating fallback."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            "strict": True,
        },
    }


# Helper function to check for TypedDict, supporting older typing_extensions
_TYPED_DICT_META_TYPES = []
try:
    from typing import _TypedDictMeta  # type: ignore

    _TYPED_DICT_META_TYPES.append(_TypedDictMeta)
except ImportError:
    pass
try:
    from typing_extensions import (
        _TypedDictMeta as _TypedDictMetaExtensions,  # type: ignore
    )

    _TYPED_DICT_META_TYPES.append(_TypedDictMetaExtensions)
except (ImportError, AttributeError):
    pass


def _is_typeddict(tool_type: type) -> bool:
    if not _TYPED_DICT_META_TYPES:
        # Fallback if metaclasses can't be imported, check for common TypedDict attributes
        return hasattr(tool_type, "__required_keys__") or hasattr(
            tool_type, "__optional_keys__"
        )
    return isinstance(tool_type, tuple(_TYPED_DICT_META_TYPES))


def _normalize_tool_call(tc: dict) -> dict:
    """Defensive normalization for tool call dicts.

    - Ensures 'id' is a non-empty string (generates uuid if missing/empty)
    - Ensures 'name' is a string (fallback to 'unknown_tool')
    - Ensures 'args' is a dict (parses string as JSON, else wraps as {'value': ...})
    - Always sets 'type' to 'function'
    - Logs a warning for any repair.
    """
    logger = logging.getLogger(__name__)
    tool_call = dict(tc)  # shallow copy

    # ID
    tool_call_id = tool_call.get("id")
    if (
        not tool_call_id
        or not isinstance(tool_call_id, str)
        or not tool_call_id.strip()
    ):
        tool_call_id = str(uuid.uuid4())
        logger.warning(f"Tool call missing or invalid id. Generated: {tool_call_id}")
    tool_call["id"] = tool_call_id

    # Name
    name = tool_call.get("name")
    if not name or not isinstance(name, str):
        logger.warning("Tool call missing or invalid name. Using 'unknown_tool'.")
        name = "unknown_tool"
    tool_call["name"] = name

    # Args
    args_val = tc.get("args")
    if isinstance(args_val, str):
        try:
            args_dict = json.loads(args_val)
        except Exception:
            args_dict = {"value": args_val}
    elif isinstance(args_val, dict):
        args_dict = args_val
    else:
        args_dict = {"value": str(args_val)}
    tool_call["args"] = args_dict

    # Type
    tool_call["type"] = "function"

    return tool_call


def _parse_textual_tool_args(args_str: Optional[str]) -> dict[str, Any]:
    """Parses a string of arguments like 'key="value", key2=value2' into a dict.

    Handles JSON-like structures if possible, otherwise falls back to regex parsing.
    """
    if not args_str or not args_str.strip():
        return {}

    # Attempt to parse as JSON first, as it's the most robust
    try:
        # Ensure outer braces for valid JSON object if it's just key:value pairs
        # This handles cases like '{"key": "value"}' or even 'key: "value"' if it's valid enough
        potential_json_str = args_str
        if not potential_json_str.startswith("{") or not potential_json_str.endswith(
            "}"
        ):
            # Basic check if it looks like a raw string needing to be quoted for a single arg tool
            if (
                ":" not in potential_json_str
                and "=" not in potential_json_str
                and '"' not in potential_json_str
            ):
                # It might be a single unquoted string for a tool that takes one arg named e.g. "query" or "input"
                # We can't know the arg name here, so we'll wrap it with a default key like "value" or let regex handle it
                pass  # Let regex try or handle as single value if JSON fails

        # More robust JSON parsing attempt
        try:
            loaded_args = json.loads(potential_json_str)
            if isinstance(loaded_args, dict):
                return loaded_args
        except json.JSONDecodeError:
            # If it's not a valid JSON object string, try adding braces
            if not potential_json_str.startswith("{"):
                potential_json_str = "{" + potential_json_str
            if not potential_json_str.endswith("}"):
                potential_json_str = potential_json_str + "}"

            # Try to make key=value into "key":"value"
            # This is a simplified attempt and might not cover all edge cases for non-standard formats
            # Regex to find key=value or key="value" pairs.
            # It tries to match keys (alphanumeric, underscores) and values (quoted or unquoted).
            # Handles basic cases, but complex nested structures or unusual characters in unquoted values might fail.

            # Attempt 1: Try to parse key="value" or key='value' or key=value (unquoted)
            # This regex handles simple key=value, key="value", key='value'
            # It's hard to make one regex perfect for all malformed "JSON-like" strings.
            # key\s*=\s*(?:\"(.*?)\"|'(.*?)'|([^,\"'\s]+))

            # For `query="What is LangChain"` the goal is `{"query": "What is LangChain"}`

            # Simpler approach: if it's just one `key="value"`:
            single_arg_match = re.match(
                r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\"(.*?)\"\s*$", args_str
            )
            if single_arg_match:
                return {single_arg_match.group(1): single_arg_match.group(2)}

            try:
                # This is a very basic attempt to convert Python-like dict string to JSON
                # It assumes keys are unquoted or quoted with single quotes, and strings use double quotes
                # Not robust for complex cases.
                python_like_dict_str = args_str
                # Ensure keys are double-quoted
                python_like_dict_str = re.sub(
                    r"([{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
                    r'\1"\2":',
                    python_like_dict_str,
                )
                # Ensure single-quoted strings become double-quoted
                python_like_dict_str = re.sub(r"'", r'"', python_like_dict_str)

                # If it doesn't look like a dict, and json.loads failed, wrap it
                if not python_like_dict_str.strip().startswith("{"):
                    python_like_dict_str = "{" + python_like_dict_str + "}"

                loaded_args = json.loads(python_like_dict_str)
                if isinstance(loaded_args, dict):
                    return loaded_args

            except json.JSONDecodeError:
                pass  # Fall through to regex or default

    except json.JSONDecodeError:
        # Fallback to regex for simple key="value" or key=value cases if JSON fails completely
        pass

    # Fallback regex for key="value", key='value', key=value (unquoted, simple)
    # This is a simplified regex and might not capture all desired formats perfectly,
    # especially with complex values or multiple arguments not well-separated.
    args = {}
    # Regex to find key=value pairs where value can be quoted or unquoted.
    # It captures: 1=key, 2=double-quoted value, 3=single-quoted value, 4=unquoted value
    # SIMPLIFIED REGEX TO GET PAST SYNTAX ERROR
    pattern = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\"(.*?)\""
    )  # Focus on key="value"
    for match in pattern.finditer(args_str):
        key = match.group(1)
        value = match.group(2)  # Only one value group now
        # Try to convert to int/float/bool if applicable, otherwise keep as string
        if value.lower() == "true":
            args[key] = True

    if not args and args_str:  # If regex found nothing but there was an args_str
        # This could be a single string argument for a tool that expects e.g. a query.
        # We can't know the schema here, so we return it with a default key 'value'
        # or the user of this function has to be aware.
        # For the specific case of `query=\"What is LangChain\"` failing with the above,
        # this won't help if it's not parsed by the regex.

        # REMOVING PROBLEMATIC FALLBACK LOGIC TO ISOLATE SYNTAX ERROR
        # # One last attempt for `key=\"value\"` where key might have spaces (not ideal for keys)
        # # or for just a single value that should be the query
        # # The provided `query=\"What is LangChain\"` should be caught by the regex.
        # # The issue is that the value has a quote inside: `\"What is LangChain\"`
        # # Let's refine the regex slightly for quoted values that might contain escaped quotes
        # # This is hard without knowing the exact escaping rules.
        #
        # # If all else fails and the string contains 'query=', assume it's the primary argument for Tavily
        # # Corrected quote escaping
        # query_match = re.search(r'query=(?:"(.*?)"|'(.*?)'|([^,\s]+))', args_str, re.IGNORECASE)
        # if query_match:
        #     query_val = next((g for g in query_match.groups() if g is not None), None)
        #     if query_val:
        #         return {"query": query_val}

        # Default if no parsing worked but string is not empty
        return {"value": args_str}

    return args


def _lc_tool_to_llama_tool_param(
    lc_tool: Union[dict[str, Any], type[BaseModel], Callable, BaseTool],
) -> completion_create_params.Tool:
    """Converts a LangChain tool/schema to a Llama API tool parameter structure using langchain_core's convert_to_openai_tool."""
    tool_name_for_log = str(lc_tool)
    if hasattr(lc_tool, "__name__") and not isinstance(lc_tool, dict):
        tool_name_for_log = lc_tool.__name__
    elif isinstance(lc_tool, dict) and (lc_tool.get("name") or lc_tool.get("title")):
        tool_name_for_log = str(lc_tool.get("name") or lc_tool.get("title"))

    try:
        # convert_to_openai_tool is robust and handles Pydantic models, dicts (JSON schema), TypedDicts, functions, BaseTools.
        openai_tool_dict = convert_to_openai_tool(lc_tool)

        if "function" not in openai_tool_dict or not isinstance(
            openai_tool_dict["function"], dict
        ):
            raise ValueError(
                "convert_to_openai_tool did not return the expected structure with a 'function' dict."
            )

        func_dict = openai_tool_dict["function"]

        # Ensure parameters schema is valid at a basic level for Llama
        final_parameters = func_dict.get("parameters", {})
        if not isinstance(final_parameters, dict):
            logger.warning(
                f"Parameters for tool '{func_dict.get('name')}' were not a dict: {final_parameters}. Defaulting to empty schema."
            )
            final_parameters = {"type": "object", "properties": {}}

        # Ensure basic structure and simplify Optional fields for Llama API
        if not final_parameters:  # If parameters was an empty dict
            final_parameters = {"type": "object", "properties": {}}
        elif "type" not in final_parameters:  # type key is expected by Llama
            final_parameters["type"] = "object"

        if final_parameters.get("type") == "object":
            if "properties" not in final_parameters:
                final_parameters["properties"] = {}
            else:
                # Simplify Optional fields (anyOf with null)
                properties = final_parameters.get("properties")
                if isinstance(properties, dict):
                    for prop_name, prop_schema in list(
                        properties.items()
                    ):  # Iterate over a copy
                        if isinstance(prop_schema, dict) and "anyOf" in prop_schema:
                            any_of_options = prop_schema.get("anyOf", [])
                            non_null_schemas = [
                                opt
                                for opt in any_of_options
                                if isinstance(opt, dict) and opt.get("type") != "null"
                            ]
                            has_null = any(
                                isinstance(opt, dict) and opt.get("type") == "null"
                                for opt in any_of_options
                            )

                            if has_null and len(non_null_schemas) == 1:
                                # This is a simple Optional[Type], e.g., Optional[str]
                                simplified_schema = non_null_schemas[0].copy()

                                # Preserve description and title from the original prop_schema
                                # if they existed at the top level of the property definition.
                                # Pydantic's model_json_schema often puts description inside the
                                # anyOf subschema for the non-null type, which is good.
                                # This ensures if it was at the outer level, it's not lost.
                                if (
                                    "description" in prop_schema
                                    and "description" not in simplified_schema
                                ):
                                    simplified_schema["description"] = prop_schema[
                                        "description"
                                    ]
                                if (
                                    "title" in prop_schema
                                    and "title" not in simplified_schema
                                ):
                                    simplified_schema["title"] = prop_schema["title"]
                                # Other schema keywords like 'enum', 'format', 'default' from non_null_schemas[0]
                                # are carried over by .copy().

                                properties[prop_name] = simplified_schema
                                logger.debug(
                                    f"Simplified Optional schema for field '{prop_name}' from {prop_schema} to {simplified_schema}"
                                )
                            elif has_null and len(non_null_schemas) > 1:
                                # This is Optional[Union[TypeA, TypeB, ...]]
                                logger.warning(
                                    f"Field '{prop_name}' is an Optional Union of multiple non-null types: {prop_schema}. "
                                    f"This complex structure might not be fully supported by Llama API. Leaving as is."
                                )
                            # If not has_null, or no non_null_schemas, or other complex cases,
                            # leave it as is. convert_to_openai_tool might have done its best.

        # Ensure 'parameters' is not None before passing to Tool TypedDict
        if final_parameters is None:  # Should be an empty dict if no params
            final_parameters = {"type": "object", "properties": {}}

        return completion_create_params.Tool(
            type="function",
            function={
                "name": str(func_dict.get("name")),
                "description": str(
                    func_dict.get("description", "")
                ),  # Ensure description is a string
                "parameters": final_parameters,
                "strict": True,
            },
        )

    except Exception as e:
        logger.error(
            f"Failed to convert tool '{tool_name_for_log}' (type: {type(lc_tool)}) to Llama tool format: {e}. Creating a dummy tool.",
            exc_info=True,
        )
        dummy_name = f"fallback_{tool_name_for_log.replace(' ', '_')[:20]}_{str(uuid.uuid4())[:4]}"
        return completion_create_params.Tool(
            type="function",
            function={
                "name": dummy_name,
                "description": f"Fallback due to conversion error for {tool_name_for_log}",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        )
