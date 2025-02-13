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
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.prompts import PromptTemplate

class PGConnection(TypedDict):
    user: str
    password: str
    host: str
    port: int
    dbname: str

class Hector:

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
        try:
            self.cursor.execute(create_tables_sql)
            logging.info("Initial tables created")
        except:
            logging.info("Initial tables already exists")

            

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



class GraphRag:

    def __init__(self, hector: Hector, llm):
        
        self.hector = hector
        self.llm = llm
        self.llm_transformer = LLMGraphTransformer(llm=llm)
        self.graph = NetworkxEntityGraph()

        self.entity_creation_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                You are an entity extraction assistant. Your task is to extract and return only named entities (people, organizations, events, places) from the given text.
                Instructions:
                Extract only relevant named entities such as people, organizations, events, and places.
                Do not include common nouns, generic words, or unnecessary phrases.
                Maintain the exact spelling and capitalization as in the input text.
                Output format: If there is one entity, return:
                Entity1
                If there are multiple entities, return:
                Entity1, Entity2, Entity3
                Do not add extra words, explanations, or change the format in any way.
                Examples:
                Input: "Racer World Championship is happening and Ankit Kokane and Prathmesh Muthkure are participating."
                Output: Racer World Championship, Ankit Kokane, Prathmesh Muthkure

                Input: "Who is Alice Johnson and it's a very sunny day so Swapnil Shinde is very happy about it."
                Output: Alice Johnson, Swapnil Shinde

                Input: "Ethereum Foundation and Vitalik Buterin are working on new blockchain innovations."
                Output: Ethereum Foundation, Vitalik Buterin

                Now, extract named entities from the following text:
                "{question}"
            """
        )

        self._create_graph_table()

    def _create_graph_table(self):

        sql = """
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE nodes (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT UNIQUE NOT NULL,
                type TEXT
            );

            CREATE TABLE edges (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_id UUID REFERENCES nodes(id),
                target_id UUID REFERENCES nodes(id),
                relationship TEXT NOT NULL
            );
        """
        try:
            self.hector.cursor.execute(sql, (self.hector.embedding_dimension, ))
            logging.info("Graph entity tables created")
        except:
            logging.info("Graph entity tables already exists")



    def add_documents(self, documents: List[Document]):

        """
            Adds documents to vector store then updates the instance graph entity
        """

        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)

        nodes = [(node.id, node.type) for node in graph_documents[0].nodes]

        psycopg2.extras.execute_values(
            self.hector.cursor, 
            "INSERT INTO nodes (name, type) VALUES %s ON CONFLICT (name) DO NOTHING", 
            nodes
        )

        logging.info("All Nodes Inserted successfully!")


        sql = """
            INSERT INTO edges (source_id, target_id, relationship)
            VALUES (
                (SELECT id FROM nodes WHERE name = %s),
                (SELECT id FROM nodes WHERE name = %s),
                %s
            );
        """

        edges = [
            (
                edge.source.id,
                edge.target.id,
                edge.type
            ) for edge in graph_documents[0].relationships
        ]

        psycopg2.extras.execute_batch(self.hector.cursor, sql, edges)
        logging.info("All Edges Inserted successfully!")

    def load_graph(self, batch_size: int = 1000):


        # load nodes 
        self.hector.cursor.execute("SELECT name FROM nodes;")

        while True:

            rows = self.hector.cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                self.graph.add_node(row[0])

        logging.info("All Nodes loaded")
    
        self.hector.cursor.execute("""
            SELECT 
                n1.name AS source_name, 
                n2.name AS target_name, 
                e.relationship
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id;
        """)

        while True:

            rows = self.hector.cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                self.graph._graph.add_edge(
                    row[0],
                    row[1],
                    relation=row[2]
                )

        logging.info("All Edges loaded")
        logging.info("Graph Loaded!")

    def get_relevant_documents(self, query: str):
        question = self.entity_creation_prompt.format(question=query)
        response = self.llm(question)
        entities = response.content.split(",")
        print(type(response.content))
        print(entities)
        print(len(entities))
        if len(entities) == 0:
            entities = response.content

        information_list = sum([self.graph.get_entity_knowledge(entity.strip()) for entity in entities], [])
        documents = [Document(page_content=information) for information in information_list]
        return documents
    
    def print_graph(self):
        logging.info(self.graph._graph.edges)
    

if __name__ == "__main__":  

    from dotenv import load_dotenv
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import ChatOpenAI

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

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    collection_name = "new_collection_1"
    hv = Hector(connection, embeddings_model, collection_name, {})

    gr = GraphRag(hector=hv, llm=llm)

    # rank1 = hv.kw_search_with_ranking("What is Decentralized AI ?", 4)
    # rank2 = hv.similarity_search_with_ranking("What is Decentralized AI ?", 4)
    # combined = hv.reciprocal_rank_fusion(rank1,rank2)

    # print(combined)

    # a = {'7f739b6a-0614-4f12-a7d3-646d0b00648d': 1}

    # results = hv._fetch_documents(list(a.keys()))
    # print(results)
    # hv.add_documents(documents)
    # gr.add_documents(documents)
    gr.load_graph()
    gr.print_graph()
    while True:
        query = input("Enter Question: ")

        documents = gr.get_relevant_documents(query)
        print(documents)



