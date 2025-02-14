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
pip install hector-rag
```

Using Poetry:
```bash
poetry add hector-rag
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

```python
from hector_rag import HectorRAG
from hector_rag.retrievers import SimilarityRetriever, KeywordRetriever

# Initialize Hector RAG
rag = HectorRAG(
    connection_string="postgresql://user:password@localhost:5432/db_name"
)

# Create retrievers
similarity_retriever = SimilarityRetriever()
keyword_retriever = KeywordRetriever()

# Add retrievers to pipeline
rag.add_retriever(similarity_retriever)
rag.add_retriever(keyword_retriever)

# Perform retrieval
results = rag.retrieve("Your query here")
```

## Advanced Usage

### Combining Multiple Retrievers with RRF

```python
from hector_rag import HectorRAG
from hector_rag.retrievers import SimilarityRetriever, GraphRetriever
from hector_rag.fusion import ReciprocralRankFusion

# Initialize retrievers
retrievers = [
    SimilarityRetriever(weight=0.4),
    GraphRetriever(weight=0.6)
]

# Create fusion pipeline
rag = HectorRAG(
    retrievers=retrievers,
    fusion_method=ReciprocralRankFusion()
)

# Get fused results
results = rag.retrieve("Your query here")
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