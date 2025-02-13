import os
import psycopg2
import json 
import logging
import psycopg2.extras

from uuid import uuid4

from typing import List, TypedDict, Dict, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

class PGConnection(TypedDict):
    user: str
    password: str
    host: str
    port: int
    dbname: str

class HybridPGVector:

    def __init__(self, connection: PGConnection, embeddings: Embeddings, collection_name: str, collection_metada: dict) -> None:

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

        self._init_tables()
        self._create_collection()

    def similarity_search(self, query: str, document_limit: int) -> List[Document]:
        
        doc_ranking = self.similarity_search_with_ranking(query, document_limit)
        docs = doc_ranking.keys()
        return self._fetch_documents(list(docs))
    
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
    

    def kw_search(self, query: str, document_limit: int) -> List[Document]:
        
        doc_ranking = self.kw_search_with_ranking(query, document_limit)
        docs = doc_ranking.keys()
        return self._fetch_documents(list(docs))
        

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


    
    def reciprocal_rank_fusion(self, rank1: List[Dict[str,int]], rank2: List[Dict[str,int]], filter_docs: Optional[int] = None) -> List[Document]:
        texts = self.reciprocal_rank_fusion_ranking(rank1, rank2, filter_docs)
        return self._fetch_documents(texts)

    def reciprocal_rank_fusion_ranking(self, rank1: List[Dict[str,int]], rank2: List[Dict[str,int]], filter_docs: Optional[int] = None) -> List[str]:

        """
            Reco[rpcal rank fusion combines and rerank documents obtained from different searching method
            uses formula:
                score(d) = SUMMATION ( 1 / ( k + r(d) ) )
            where:
                K = parameter constant, originally k=60 used in research paper
                r(d) = rank of the document d

            combining 2 document ranking:
                rrf_score =  1 / (k + rank1) + 1 / (k + rank2) # for document["<document_uid>"] of rank1 and rank2
        """

        if filter_docs is not None and filter_docs > max(len(rank1), len(rank2)):
            filter_docs = None

        combined_score_uid_info = {}
        max_document_rank1 = len(rank1) + 1
        max_document_rank2 = len(rank2) + 1


        for key in rank1:
            combined_score_uid_info[key] = self._reciprocal_rank_fusion_formula(rank1[key], rank2.get(key, max_document_rank1)) # use max rank as len of rank1 list to remove unfair advantage in long lists

        for key in rank2:
            combined_score_uid_info[key] = self._reciprocal_rank_fusion_formula(rank2[key], rank1.get(key, max_document_rank2)) # use max rank as len of rank2 list to remove unfair advantage in long lists

        sorted_combined_rank = [k for k, v in sorted(combined_score_uid_info.items(), key=lambda item: item[1], reverse=True)] # Get sorted keys according to rank
        
        if filter_docs == None:
            return sorted_combined_rank
        else:
            return sorted_combined_rank[:filter_docs]

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


    @staticmethod
    def rank_keywords(kw_score: List[int]) -> List[int]:
        
        """
        Ranks keywords based on their scores in descending order.

        Args:
            kw_score (list): List of keyword scores.

        Returns:
            list: Ranked indices starting from 1.
        """

        sorted_indices = sorted(range(len(kw_score)), key=lambda i: kw_score[i], reverse=True)
        ranking = [0] * len(kw_score)
        
        for rank, idx in enumerate(sorted_indices, start=1):
            ranking[idx] = rank

        return ranking
    

    def _init_tables(self):
        """
        If tables don't exist, initializes `langchain_pg_collection` 
        and `langchain_pg_embedding` with appropriate datatypes.
        """

        check_table_pg_collection_sql = """
            SELECT 
                EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'langchain_pg_collection'),
                EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'langchain_pg_embedding');
        """
        self.cursor.execute(check_table_pg_collection_sql)
        check_table_pg_collection_exists, check_table_pg_embedding_exists = self.cursor.fetchone()

        logging.info(f"table langchain_pg_collection exists = {check_table_pg_collection_exists}")
        logging.info(f"table langchain_pg_embedding exists = {check_table_pg_embedding_exists}")

        if check_table_pg_collection_exists and check_table_pg_embedding_exists:
            return  # Tables already exist, no need to create them

        create_tables_sql = f"""
            BEGIN;
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
                embedding VECTOR({self.embedding_dimension}),
                document VARCHAR(1000),
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

        self.cursor.execute(create_tables_sql)
        logging.info("Tables created or already exist")

            

    def _create_collection(self):

        sql = """
           SELECT CASE 
                WHEN EXISTS (
                    SELECT 1 
                    FROM langchain_pg_collection 
                    WHERE name = %s
                )
                THEN uuid::text
                ELSE 'false'
            END
            FROM langchain_pg_collection
            LIMIT 1;
        """

        self.cursor.execute(sql, (self.collection_name,))
        exists = self.cursor.fetchone()


        if exists:
            self.collection_uuid = exists[0]
            logging.info(f"Collection {self.collection_name} exists with uuid {exists[0]}")
            return

        sql = """
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
            VALUES (uuid_generate_v4(), %s, %s) RETURNING uuid;
        """

        cmetadata = json.dumps(self.collection_metada)
        self.cursor.execute(sql, (self.collection_name, cmetadata))
        collection_uuid = self.cursor.fetchone()[0]

        self.collection_uuid = collection_uuid
        logging.info(f"Collection {self.collection_name} created")


    def _fetch_documents(self, document_uids: List[str]) -> List[Document]:

        sql = "SELECT cmetadata,document FROM langchain_pg_embedding WHERE uuid = ANY(%s::uuid[]);"
        self.cursor.execute(sql, (document_uids,))

        text_documents = self.cursor.fetchall()        
        documents = []

        for t_doc in text_documents:
            documents.append(
                Document(
                    metadata = t_doc[0],
                    page_content = t_doc[1]
                )
            )

        return documents

    def _reciprocal_rank_fusion_formula(self, rank1: int, rank2: int) -> int:
        return 1 / (self.rrf_constant + rank1) + 1 / (self.rrf_constant + rank2)



if __name__ == "__main__":  

    from dotenv import load_dotenv
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter

    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    )

    text_loader = TextLoader(file_path="./assets/dataset.txt")
    documents = text_loader.load()

    text_splitter = CharacterTextSplitter(
        separator=".",  # Split on double newlines (adjust as needed)
        chunk_size=500,     # Adjust chunk size as per requirement
        chunk_overlap=50    # Overlapping tokens for better context
    )

    documents = text_splitter.split_documents(documents)


    print(os.getenv("DB_USER"))
    connection = {
        "user":os.getenv("DB_USER"),
        "password":os.getenv("DB_PASSWORD"),
        "host":os.getenv("DB_HOST"),
        "port":os.getenv("DB_PORT"),
        "dbname":os.getenv("DB_NAME")
    }

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    collection_name = "new_collection_1"
    hv = HybridPGVector(connection, embeddings_model, collection_name, {})


    rank1 = hv.kw_search_with_ranking("What is Decentralized AI ?", 4)
    rank2 = hv.similarity_search_with_ranking("What is Decentralized AI ?", 4)
    combined = hv.reciprocal_rank_fusion(rank1,rank2)

    print(combined)

    # a = {'7f739b6a-0614-4f12-a7d3-646d0b00648d': 1}

    # results = hv._fetch_documents(list(a.keys()))
    # print(results)
    # hv.add_documents(documents)



