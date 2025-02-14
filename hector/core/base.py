from typing import TypedDict, List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class PGConnection(TypedDict):
    user: str
    password: str
    host: str
    port: int
    dbname: str


class BaseRetriever(ABC):

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
