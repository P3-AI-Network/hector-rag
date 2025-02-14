import logging
import json 

from typing import List, Dict
from psycopg2.extensions import cursor
import psycopg2.extras
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.base import fetch_documents, rank_keywords
from core.base import BaseRetriever


class SimilarityRetriever(BaseRetriever):

    def __init__(self, cursor: cursor, embeddings: Embeddings, embeddings_dimention: int):
        self.cursor = cursor
        self.embeddings = embeddings
        self.embeddings_dimention = embeddings_dimention


    def get_relevant_documents(self, query: str, document_limit: int) -> List[Document]:
        
        doc_ranking = self.similarity_search_with_ranking(query, document_limit)
        docs = doc_ranking.keys()
        return fetch_documents(list(docs))
    
    def similarity_search_with_ranking(self, query: str, document_limit: int) -> Dict[str, int]:

        score_uid_info = self.similarity_search_with_score(query, document_limit)

        scores = [] # [<score_doc_1>, <score_doc_2>]

        for score in score_uid_info.values():
            scores.append(score)

        ranking = rank_keywords(scores)

        for index, key in enumerate(score_uid_info):
            score_uid_info[key] = ranking[index]
        
        return score_uid_info
        
    
    def similarity_search_with_score(self, query: str, document_limit: int) -> Dict[str, float]:

        sql = """
            WITH semantic_search AS (
                SELECT uuid, document, (1 - (embedding <#> %s::vector)) AS similarity
                FROM langchain_pg_embedding
                ORDER BY similarity DESC
                LIMIT %s
            )
            SELECT * FROM semantic_search;
        """

        single_vector = self.embeddings.embed_query(query)
        self.cursor.execute(sql, (single_vector,document_limit))

        results = self.cursor.fetchall()

        score_uid_info = {} # {'<uuid>': <score>}

        for row in results:
            score_uid_info[row[0]] = row[-1]
        
        return score_uid_info
    
    def add_documents(self, documents: List[Document]):

        sql = """
            INSERT INTO langchain_pg_embedding (
                collection_id, embedding, document, cmetadata, custom_id
            ) VALUES %s;
        """

        text_docs = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(text_docs)

        insert_data = []

        for doc, embedding in zip(documents, embeddings):
            insert_data.append((self.collection_uuid, embedding, doc.page_content, json.dumps(doc.metadata), str(uuid4())))

        psycopg2.extras.execute_values(self.cursor, sql, insert_data, template="(%s, %s, %s, %s, %s)")
        logging.info("Documents Inserted!")
    
    def init_tables(self):
         
        """
            If tables don't exist, initializes `langchain_pg_collection` 
            and `langchain_pg_embedding` with appropriate datatypes.
        """


        create_tables_sql = f"""
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                cmetadata JSON
            );
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                collection_id UUID NOT NULL,
                embedding VECTOR({self.embedding_dimension}),
                document VARCHAR(1000),
                cmetadata JSON,
                custom_id VARCHAR,
                uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4()
            );
        """
        try:
            self.cursor.execute(create_tables_sql)
            logging.info("Initial tables created")
        except:
            logging.info("Initial tables already exists")
    