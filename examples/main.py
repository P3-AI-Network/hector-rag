import os
import logging
import psycopg2
# from dotenv import load_dotenv
from hector_rag import Hector
from hector_rag.retrievers import GraphRetriever, SemanticRetriever, KeywordRetriever

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

# load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


connection = {
        "user":os.getenv("DB_USER"),
        "password":os.getenv("DB_PASSWORD"),
        "host":os.getenv("DB_HOST"),
        "port":os.getenv("DB_PORT"),
        "dbname":os.getenv("DB_NAME")
}

print(connection)

connection_obj = psycopg2.connect(
    user=connection['user'],
    password=connection['password'],
    host=connection['host'],
    port=connection['port'],
    dbname=connection['dbname']
)
connection_obj.autocommit = True
cursor = connection_obj.cursor()

llm = ChatOpenAI(model="gpt-3.5-turbo")
logging.info("llm")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

collection_name = "new_collection_1"

hc = Hector(connection,embeddings, collection_name, {}, 1)
print(hc.embedding_dimension)
sr = SemanticRetriever(indexing=True)

# gr = GraphRetriever(llm=llm)

kw = KeywordRetriever(indexing=True)


hc.add_retriever(sr)

# IBM Security Bulletin: Privilege escalation vulnerability affects IBM Db2 Administrative Task Scheduler (CVE-2018-1711). Categorized: High Severity
# what High Severity vulnerability affects IBM Db2 Administrative Task Scheduler 


# hc.add_retriever(gr)


hc.add_retriever(kw)

# docs = hc.get_relevant_documents("What is  Decentralized AI ?", document_limit=2)

# print(docs)

# while True:

#     query = str(input("Enter query: "))
#     resp = hc.invoke(llm,query)
#     print(resp)


# sr = SemanticRetriever(cursor, embeddings, embeddings_dimension=1536, collection_name=collection_name)
# sr.init_tables()
# resp = sr.get_relevant_documents("What is Fetch Ai ?", 10)

# print(resp)
