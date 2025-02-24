import logging
import json 

from typing import List, Dict, Optional
from psycopg2.extensions import cursor
import psycopg2.extras
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hector_rag.utils.base import fetch_documents
from hector_rag.core.base import BaseRetriever
from hector_rag.fusion.reciprocal_rank_fusion import ReciprocralRankFusion


class SemanticRetriever(BaseRetriever, ReciprocralRankFusion):

    def __init__(
            self, 
            cursor: Optional[cursor] = None, 
            embeddings: Optional[Embeddings] = None, 
            embeddings_dimension: Optional[int] = None, 
            collection_name: Optional[str] = None,
            indexing: bool = False,
            **kwargs    
        ):
        self.cursor = cursor
        self.embeddings = embeddings
        self.embeddings_dimension = embeddings_dimension
        self.collection_name = collection_name
        self.collection_uuid = None
        self.collection_metadata = {}

        self.indexing = indexing


    def get_relevant_documents(self, query: str, document_limit: int) -> List[Document]:
        
        logging.info("Semantic Search Started!")
        doc_ranking = self.similarity_search_with_ranking(query, document_limit)
        doc_ids = doc_ranking.keys()
        docs = fetch_documents(self.cursor,list(doc_ids))[:document_limit]
        logging.info("Semantic Search Started!")
        return docs
    
    def similarity_search_with_ranking(self, query: str, document_limit: int) -> Dict[str, int]:

        score_uid_info = self.similarity_search_with_score(query, document_limit)

        scores = [] # [<score_doc_1>, <score_doc_2>]

        for score in score_uid_info.values():
            scores.append(score)

        ranking = self.rank_keywords(scores)

        for index, key in enumerate(score_uid_info):
            score_uid_info[key] = ranking[index]
        
        return score_uid_info
        
    
    def similarity_search_with_score(self, query: str, document_limit: int) -> Dict[str, float]:

        sql = """
            WITH semantic_search AS (
                SELECT uuid, document, (1 - (embedding <#> %s::vector)) AS similarity
                FROM langchain_pg_embedding
                WHERE collection_id=%s
                ORDER BY similarity DESC
                LIMIT %s
            )
            SELECT * FROM semantic_search;
        """

        single_vector = self.embeddings.embed_query(query)
        self.cursor.execute(sql, (single_vector, self.collection_uuid, document_limit))

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
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                cmetadata JSON
            );
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                collection_id UUID NOT NULL,
                embedding VECTOR({self.embeddings_dimension}),
                document TEXT NOT NULL,
                cmetadata JSON,
                custom_id VARCHAR,
                uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4()
            );
        """

        indexing_sql = """
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw 
            ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 200);
        """

        # Table Creation
        try:
            self.cursor.execute(create_tables_sql)
            logging.info("Semantic Retriever: Initial tables created")
            self._create_collection()

        except Exception as e:
            print("DB Error: ", e)
            logging.info("Initial tables already exists")

        # Index Creation
        try:
            if self.indexing:
                logging.info("Semantic Retriever: Creating Index! This might take a while depending on your database rows")
                self.cursor.execute(indexing_sql)
                logging.info("Semantic Retriever: Index Created")
        except Exception as e:
            print("DB Error: ", e)
            logging.info("Semantic Retriever: Index Creation Failed")
        
        # Fetch Collection ID
        try:
            self.cursor.execute("SELECT uuid from langchain_pg_collection WHERE name=%s", (self.collection_name,))
            result = self.cursor.fetchone()[0]
            self.collection_uuid = result

        except Exception as e:
            print("DB Error: ", e)
            logging.info("Semantic Retriever: Collection does not exists")
    

    def _create_collection(self):

        sql = """
           SELECT COALESCE(
                (SELECT uuid::text FROM langchain_pg_collection WHERE name = %s LIMIT 1), 
                'false'
            );
        """

        self.cursor.execute(sql, (self.collection_name,))
        exists = self.cursor.fetchone()[0]


        if not exists == 'false':
            self.collection_uuid = exists
            logging.info(f"Collection {self.collection_name} exists with uuid {exists[0]}")
            return

        sql = """
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
            VALUES (uuid_generate_v4(), %s, %s) RETURNING uuid;
        """

        cmetadata = json.dumps(self.collection_metadata)
        self.cursor.execute(sql, (self.collection_name, cmetadata))
        collection_uuid = self.cursor.fetchone()[0]

        self.collection_uuid = collection_uuid
        logging.info(f"Collection {self.collection_name} created")

