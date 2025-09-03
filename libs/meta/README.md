# langchain-meta

This package contains the LangChain integrations for [Meta](https://llama.com/) through their [APIs](https://llama.developer.meta.com?utm_source=partner-langchain&utm_medium=readme).

## Installation and Setup

- Install the LangChain partner package

```bash
pip install -U langchain-meta
```

- Get your Llama api key from the [Meta](https://llama.developer.meta.com?utm_source=partner-langchain&utm_medium=readme) and set it as an environment variable (`LLAMA_API_KEY`)

## Chat Completions

This package provides the `ChatLlama` class for interfacing with Llama chat models.

### Basic Usage

```python
from langchain_meta import ChatLlama
from langchain_core.messages import HumanMessage

# Initialize the chat model
llm = ChatLlama(model="Llama-3.3-8B-Instruct")

# Basic chat
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

### Advanced Features

#### Tool Calling

```python
from langchain_meta import ChatLlama
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny."

# Bind tools to the model
llm = ChatLlama(model="Llama-3.3-8B-Instruct")
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in San Francisco?")
print(response.tool_calls)
```

#### Streaming

```python
# Stream responses
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

#### Structured Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Get structured responses
llm_structured = llm.with_structured_output(Person)
result = llm_structured.invoke("Tell me about a person named Alice who is 30")
print(f"Name: {result.name}, Age: {result.age}")
```

## Enhanced Backend (Optional)

For enhanced tool calling and streaming capabilities, you can use the native backend:

```python
# Use enhanced backend for better tool calling
llm = ChatLlama(model="Llama-3.3-8B-Instruct", use_native_client=True)
```

The enhanced backend provides:

- Improved tool calling reliability
- Better streaming performance
- Enhanced error handling

## Environment Variables

```bash
# Required
export LLAMA_API_KEY="your-api-key"

# Optional (defaults to OpenAI-compatible endpoint)
export LLAMA_API_BASE="https://api.llama.com/compat/v1/"
```

## API Models

| Model ID                                 | Input context length | Output context length | Input Modalities | Output Modalities |
| ---------------------------------------- | -------------------- | --------------------- | ---------------- | ----------------- |
| `Llama-4-Scout-17B-16E-Instruct-FP8`     | 128k                 | 4028                  | Text, Image      | Text              |
| `Llama-4-Maverick-17B-128E-Instruct-FP8` | 128k                 | 4028                  | Text, Image      | Text              |
| `Llama-3.3-70B-Instruct`                 | 128k                 | 4028                  | Text             | Text              |
| `Llama-3.3-8B-Instruct`                  | 128k                 | 4028                  | Text             | Text              |
