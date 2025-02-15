from typing import TypedDict, List, Optional
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg2.extensions import cursor

class PGConnection(TypedDict):
    user: str
    password: str
    host: str
    port: int
    dbname: str


class BaseRetriever(ABC):

    def __init__(
            self,
            cursor: Optional[cursor], 
            embeddings: Optional[Embeddings] = None, 
            embeddings_dimension: Optional[int] = None, 
            collection_uuid: Optional[str] = None,
            llm: Optional[any] = None,
            **kwargs
        ):
        
        self.cursor = cursor
        self.embeddings = embeddings
        self.embeddings_dimension = embeddings_dimension
        self.collection_uuid = collection_uuid
        self.llm = llm

    @abstractmethod
    def get_relevant_documents(self,query: str, document_limit: int) -> List[Document]:
        """Perform base search and retrieve documents"""
        pass

    @abstractmethod
    def init_tables(self) -> None:
        """Initialize search vector tables in pgsql"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to vector store then updates the instance graph entity"""
        pass
    
    def load(self):
        pass
