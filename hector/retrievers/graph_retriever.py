import logging
import psycopg2.extras

from psycopg2.extensions import cursor
from typing import List

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.prompts import PromptTemplate

from prompts.templates import EXTITY_EXTRACTION_PROMPT_TEMPLATE

from hector.core.base import BaseRetriever

class GraphRetriever(BaseRetriever):

    def __init__(self, cursor: cursor, llm, weight: float):
        
        self.cursor = cursor
        self.llm = llm
        self.llm_transformer = LLMGraphTransformer(llm=llm)
        self.graph = NetworkxEntityGraph()

        self.weight = weight

        self.entity_creation_prompt = EXTITY_EXTRACTION_PROMPT_TEMPLATE
        self.init_tables()

    def init_tables(self):

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
            self.cursor.execute(sql)
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
            self.cursor, 
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

        psycopg2.extras.execute_batch(self.cursor, sql, edges)
        logging.info("All Edges Inserted successfully!")

    def load_graph(self, batch_size: int = 1000):


        # load nodes 
        self.cursor.execute("SELECT name FROM nodes;")

        while True:

            rows = self.cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                self.graph.add_node(row[0])

        logging.info("All Nodes loaded")
    
        self.cursor.execute("""
            SELECT 
                n1.name AS source_name, 
                n2.name AS target_name, 
                e.relationship
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id;
        """)

        while True:

            rows = self.cursor.fetchmany(batch_size)
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

    def get_relevant_documents(self, query: str, document_limit: int):

        document_limit = int( document_limit * self.weight )

        question = self.entity_creation_prompt.format(question=query)
        response = self.llm(question)
        entities = response.content.split(",")

        if len(entities) == 0:
            entities = response.content

        information_list = sum([self.graph.get_entity_knowledge(entity.strip()) for entity in entities], [])
        documents = [Document(page_content=information) for information in information_list]
        return documents[:limit_documents]
    
    def print_graph(self):
        logging.info(self.graph._graph.edges)
    


