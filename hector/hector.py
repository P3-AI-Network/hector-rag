import psycopg2
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from hector.core import PGConnection
from hector.core.base import SearchStore

class Hector:

    def __init__(self, connection: PGConnection, embeddings: Embeddings, collection_name: str, collection_metada: dict, search_stores: List[SearchStore]) -> None:

        self.connection = psycopg2.connect(
            user=connection['user'],
            password=connection['password'],
            host=connection['host'],
            port=connection['port'],
            dbname=connection['dbname']
        )
        self.connection.autocommit = True

        self.cursor = self.connection.cursor()
        
        self.collection_name = collection_name
        self.collection_metada = collection_metada
        self.collection_uuid = None

        self.embeddings = embeddings
        self.embedding_dimension = len(embeddings.embed_query("Get Embedding"))\
            
        self.rrf_constant = 60

        self.search_stores = search_stores

        # Initialize search store tables
        for store in self.search_stores:
            store.init_tables()
    
    def add_docuemnt(self, documents: List[Document]):
        for store in self.search_stores:
            store.add_documents(documents)

    def get_relevant_documents(self, query: str, document_limit: int) -> List[Document]:

        documents = []
        per_store_limit = round(document_limit / len(self.search_stores))

        for store in self.search_stores:
            documents += store.get_relevant_documents(query, per_store_limit)

        return documents
    
    