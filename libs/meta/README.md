# langchain-meta

LangChain integration for [Meta's Model API](https://dev.meta.ai/docs/getting-started/overview/).

## Install

```bash
pip install -U langchain-meta
export MODEL_API_KEY="your-api-key"
```

## Usage

```python
from langchain_meta import ChatMetaModel

llm = ChatMetaModel(model="muse-spark-1.1")
llm.invoke("Hello!")
```
