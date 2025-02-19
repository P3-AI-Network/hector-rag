import logging
import psycopg2
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from hector_rag.core import PGConnection
from hector_rag.prompts.templates import HECTOR_QNA_PROMPT_TEMPLATE

class Hector:

    def __init__(self, connection: PGConnection, embeddings: Embeddings, collection_name: str, collection_metada: dict) -> None:

        self.connection = psycopg2.connect(
            user=connection['user'],
            password=connection['password'],
            host=connection['host'],
            port=connection['port'],
            dbname=connection['dbname']
        )
        logging.info("Vector DB connection established")
        self.connection.autocommit = True

        self.cursor = self.connection.cursor()
        
        self.collection_name = collection_name
        self.collection_metada = collection_metada
        self.collection_uuid = None

        self.embeddings = embeddings
        self.embedding_dimension = len(embeddings.embed_query("Get Embedding"))
        logging.info("Embedding dimensions added")
            
        self.rrf_constant = 60

        self.retrievers: List[BaseRetriever] = []
    
    def add_retriever(self,retriever: BaseRetriever):

        retriever.cursor = self.cursor
        retriever.embeddings = self.embeddings
        retriever.embeddings_dimension = self.embedding_dimension
        retriever.collection_uuid = self.collection_uuid
        retriever.collection_metada = self.collection_metada
        retriever.collection_name = self.collection_name
        retriever.init_tables()
        retriever.load()
        self.retrievers.append(retriever)
        logging.info(f"{retriever} added")

    def get_relevant_documents(self, query: str, document_limit: Optional[int] = 50) -> List[Document]:

        documents = []
        per_retriever_limit = round(document_limit / len(self.retrievers))

        for retriever in self.retrievers:
            documents += retriever.get_relevant_documents(query, per_retriever_limit)

        return documents
    
    def invoke(self, llm: any, query: str) -> str:

        docs = self.get_relevant_documents(query)

        docs_content = ""

        for index, doc in enumerate(docs):
            docs_content += f"\n\n Document {index} \n Content: {doc.page_content} \n Metadata: {doc.metadata} \n\n"


        formatted_prompt = HECTOR_QNA_PROMPT_TEMPLATE.format(
            context=docs_content,
            question=query
        )

        response = llm.invoke(formatted_prompt)

        return response.content
