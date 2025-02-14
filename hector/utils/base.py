from typing import List
from langchain_core.documents import Document
from psycopg2.extensions import cursor

def fetch_documents(cursor: cursor, document_uids: List[str]) -> List[Document]:

        sql = "SELECT cmetadata,document FROM langchain_pg_embedding WHERE uuid = ANY(%s::uuid[]);"
        cursor.execute(sql, (document_uids,))

        text_documents = cursor.fetchall()   
             
        documents = []

        for t_doc in text_documents:
            documents.append(
                Document(
                    metadata = t_doc[0],
                    page_content = t_doc[1]
                )
            )

        return documents
