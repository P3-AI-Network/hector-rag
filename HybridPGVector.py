import os
import psycopg2
import json 
import logging
from uuid import uuid4

from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
import psycopg2.extras

class PGConnection(TypedDict):
    user: str
    password: str
    host: str
    port: int
    dbname: str

class HybridPGVector:

    def __init__(self, connection:PGConnection, embeddings: Embeddings, collection_name: str, collection_metada: dict) -> None:

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
        self.embedding_dimension = len(embeddings.embed_query("Get Embedding"))

        self._init_tables()
        self._create_collection()

    def _init_tables(self):

        """
            If tables doesn't exists
            Initializes tables langchain_pg_collection and langchain_pg_embeddings with appropriate datatypes
        """

        check_table_pg_collection = f"""
            SELECT EXISTS (
                SELECT 1 FROM pg_tables 
                WHERE schemaname = 'public' AND tablename = 'langchain_pg_collection'
            );
        """
        self.cursor.execute(check_table_pg_collection)
        exists = self.cursor.fetchone()[0]

        logging.info(f"table langchain_pg_collection exists = {exists}")

        if not exists:
            
            create_pg_collection = """
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                CREATE TABLE langchain_pg_collection (
                    uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    cmetadata JSON
                );
            """

            self.cursor.execute(create_pg_collection)
            logging.info("table langchain_pg_collection created")
        


        check_table_pg_embedding = f"""
            SELECT EXISTS (
                SELECT 1 FROM pg_tables 
                WHERE schemaname = 'public' AND tablename = 'langchain_pg_embedding'
            );
        """


        self.cursor.execute(check_table_pg_embedding)
        exists = self.cursor.fetchone()[0]

        logging.info(f"table langchain_pg_embedding exists = {exists}")

        if not exists:
            
            create_pg_collection = f"""
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

                CREATE TABLE langchain_pg_embedding (
                    collection_id UUID NOT NULL,
                    embedding VECTOR(%s),
                    document VARCHAR(1000),
                    cmetadata JSON,
                    custom_id VARCHAR,
                    uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                );

                ALTER TABLE langchain_pg_embedding ADD COLUMN search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
            """

            self.cursor.execute(create_pg_collection, (self.embedding_dimension,))
            logging.info("table langchain_pg_embedding created")
    
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

        logging.info(f"Collection {self.collection_name} exists with uuid {exists[0]}")

        if exists:
            self.collection_uuid = exists[0]
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
        logging.info("Document Inserted!")



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



    hv.add_documents(documents)

