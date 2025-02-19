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

class KeywordRetriever(BaseRetriever, ReciprocralRankFusion):

    def __init__(
            self, 
            cursor: Optional[cursor] = None, 
            embeddings: Optional[Embeddings] = None, 
            collection_name: Optional[str] = None,
            **kwargs
        ):
        self.cursor = cursor
        self.embeddings = embeddings
        self.collection_name = collection_name

    def get_relevant_documents(self, query: str, document_limit: int) -> List[Document]:
        
        logging.info("Keyword search Started!")  
        doc_ranking = self.kw_search_with_ranking(query, document_limit)
        doc_ids = doc_ranking.keys()
        docs = fetch_documents(self.cursor,list(doc_ids))[:document_limit]
        logging.info("Keyword search Complted!")  
        return docs
        

    def kw_search_with_ranking(self, query: str, document_limit: int) -> Dict[str, int]:

        score_uid_info = self.kw_search_with_score(query, document_limit)

        scores = [] # [<score_doc_1>, <score_doc_2>]

        for score in score_uid_info.values():
            scores.append(score)

        ranking = self.rank_keywords(scores)

        for index, key in enumerate(score_uid_info):
            score_uid_info[key] = ranking[index]
        
        return score_uid_info

    def kw_search_with_score(self, query: str, document_limit: int) -> Dict[str, float]:

        sql = """
            SELECT uuid, document, ts_rank_cd(search_vector, query) AS rank
            FROM langchain_pg_embedding, plainto_tsquery(%s) query
            WHERE search_vector @@ query
            ORDER BY rank DESC
            LIMIT %s;
        """

        self.cursor.execute(sql, (query,document_limit))

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
            BEGIN;
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

            -- Create langchain_pg_collection if it does not exist
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                cmetadata JSON
            );

            -- Create langchain_pg_embedding if it does not exist
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                collection_id UUID NOT NULL,
                embedding VECTOR({self.embeddings_dimension}),
                document TEXT NOT NULL,
                cmetadata JSON,
                custom_id VARCHAR,
                uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4()
            );

            -- Add search_vector column if it does not exist
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'langchain_pg_embedding' AND column_name = 'search_vector'
                ) THEN
                    ALTER TABLE langchain_pg_embedding 
                    ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
                END IF;
            END $$;

            COMMIT;
        """
        try:
            self.cursor.execute(create_tables_sql)
            logging.info("Initial tables created")

            self._create_collection()
        except Exception as e:
            print("DB Error: ", e)
            logging.info("Initial tables already exists")


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

