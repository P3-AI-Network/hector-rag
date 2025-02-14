from typing import List, Dict, Optional
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


def reciprocal_rank_fusion_formula(rank1: int, rank2: int, rrf_constant: int) -> int:
        return 1 / (rrf_constant + rank1) + 1 / (rrf_constant + rank2)

def reciprocal_rank_fusion_ranking(rank1: Dict[str,int], rank2: Dict[str,int], rrf_constant: int, filter_docs: Optional[int] = None) -> List[str]:

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
        combined_score_uid_info[key] = reciprocal_rank_fusion_formula(rank1[key], rank2.get(key, max_document_rank1), rrf_constant) # use max rank as len of rank1 list to remove unfair advantage in long lists

    for key in rank2:
        combined_score_uid_info[key] = reciprocal_rank_fusion_formula(rank2[key], rank1.get(key, max_document_rank2), rrf_constant) # use max rank as len of rank2 list to remove unfair advantage in long lists

    sorted_combined_rank = [k for k, v in sorted(combined_score_uid_info.items(), key=lambda item: item[1], reverse=True)] # Get sorted keys according to rank
    
    if filter_docs == None:
        return sorted_combined_rank
    else:
        return sorted_combined_rank[:filter_docs]


def reciprocal_rank_fusion(rank1: List[Dict[str,int]], rank2: List[Dict[str,int]], rrf_constant: Optional[int] = 60, filter_docs: Optional[int] = None) -> List[Document]:
        texts = reciprocal_rank_fusion_ranking(rank1, rank2, rrf_constant, filter_docs)
        return fetch_documents(texts)
