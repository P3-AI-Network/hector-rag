<img src="https://github.com/user-attachments/assets/05707bb7-2c4d-42f0-9397-d950308623ba" width="200" height="200" alt="Alt text">

# Hector RAG

A modular and extensible RAG (Retrieval Augmented Generation) package built on PostgreSQL vector database, offering advanced retrieval methods and fusion capabilities.

## Key Features

- Multiple RAG retrieval methods:
  - Similarity Search
  - Keyword Search
  - Graph Retrieval
  - KAG (Knowledge-Aware Graph)
- Reciprocal Rank Fusion (RRF) for combining multiple retrieval methods
- Built on PostgreSQL vector database for efficient vector storage and retrieval
- Modular architecture allowing easy integration and customization
- Advanced RAG pipeline creation capabilities

## Installation

Using pip:
```bash
pip install hector_rag
```

Using Poetry:
```bash
poetry add hector_rag
```

## Requirements

- Python >=3.10,<3.13
- PostgreSQL database
- Dependencies:
  - networkx
  - semantic-router
  - pgvector
  - sqlalchemy

## Quick Start

### Basic Usage - Using 1 pg retriever

```python
import os

from hector_rag import Hector
from hector_rag.retrievers import SimilarityRetriever, KeywordRetriever, GraphRetriever, RRFHybridRetriever
from hector_rag import Hector
from hector_rag.retrievers import GraphRetriever, SemanticRetriever, KeywordRetriever

semantic_retriever = SemanticRetriever(cursor,embeddings,embeddings_dimension=1536,collection_name=collection_name)
semantic_retriever.init_tables()
resp = semantic_retriever.get_relevant_documents(query="What is Fetch Ai ?", document_limit=10)

print(resp)
```

## Advanced Usage

### Combining Multiple Retrievers with RRF

```python
import os

from hector_rag import Hector
from hector_rag.retrievers import SimilarityRetriever, KeywordRetriever, GraphRetriever, RRFHybridRetriever
from hector_rag import Hector
from hector_rag.retrievers import GraphRetriever, SemanticRetriever, KeywordRetriever

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
collection_name = "new_collection_1"

rag = Hector(connection,embeddings, collection_name, {})

# Init all the retrievers that you want to use
semantic_retriever = SemanticRetriever()
graph_retriever = GraphRetriever(llm=llm)
keyword_retriever = KeywordRetriever()

# Add retrievers to Rag pipeline
rag.add_retriever(semantic_retriever)
rag.add_retriever(semantic_retriever)
rag.add_retriever(semantic_retriever)

# Fetch documents
docs = rag.get_relevant_documents("What is  Decentralized AI ?", document_limit=10)

# Or directly use Hector Invoke to get llm response

while True:
    query = str(input("Enter query: "))
    response = rag.invoke(llm,query)
    print(response)

```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/hector-rag.git
cd hector-rag

# Install dependencies using Poetry
poetry install
```

## Testing

```bash
poetry run pytest
```

## Documentation

For detailed documentation about each retriever type and fusion methods, please visit our [documentation page](link-to-docs).

## Contributing

Contributions are welcome! Whether it's:
- Adding new retrieval methods
- Improving existing retrievers
- Enhancing documentation
- Reporting bugs
- Suggesting features

Please feel free to submit a Pull Request or create an Issue.

## License

MIT License

## Contact

For issues and feature requests, please use the [GitHub Issues](link-to-issues) page.

Would you like me to add or modify any specific section of this README?
