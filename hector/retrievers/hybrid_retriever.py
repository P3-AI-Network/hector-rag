from typing import List, Optional

from core.base import BaseRetriever
from psycopg2.extensions import cursor

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from retrievers.keyword_retriever import KeywordRetriever
from retrievers.semantic_retriever import SemanticRetriever

from fusion.reciprocal_rank_fusion import ReciprocralRankFusion

class RRFHybridRetriever(BaseRetriever, ReciprocralRankFusion):

    def __init__(
            self, 
            cursor: cursor, 
            embeddings: Embeddings, 
            embeddings_dimention: int, 
            rrf_constant: Optional[int] = 60
        ):

        self.cursor = cursor

        self.semantic_retriever = SemanticRetriever(cursor, embeddings, embeddings_dimention)
        self.kw_retriever = KeywordRetriever(cursor, embeddings)

        ReciprocralRankFusion.__init__(rrf_constant)

    def get_relevant_documents(self, query: str, document_limit: int) -> List[Document]:
        """
            Uses Reciprocal Rank Fusion to combine semantic and keyword retrival
            returns:
                list of documents ranked in order
        """
        semantic_docs_ranking = self.semantic_retriever.similarity_search_with_ranking(query, document_limit * 2)
        kw_docs_ranking = self.kw_retriever.kw_search_with_ranking(query, document_limit * 2)
        return self.reciprocal_rank_fusion(semantic_docs_ranking, kw_docs_ranking, self.rrf_constant, document_limit)


class WeightedHybridRetriever(BaseRetriever):

    def __init__(
            self, 
            cursor: cursor, 
            embeddings: Embeddings, 
            embeddings_dimention: int, 
            semantic_search_weight: float, 
            kw_search_weight: float,
        ):

        self.cursor = cursor
        self.semantic_search_weight = semantic_search_weight
        self.kw_search_weight = kw_search_weight

        self.semantic_retriever = SemanticRetriever(cursor, embeddings, embeddings_dimention)
        self.kw_retriever = KeywordRetriever(cursor, embeddings)

    def get_relevant_documents(self, query: str, document_limit: int) -> List[Document]:
        """
            Uses Reciprocal Rank Fusion to combine semantic and keyword retrival
            returns:
                list of documents ranked in order
        """
        semantic_docs = self.semantic_retriever.get_relevant_documents(query, int( document_limit * self.semantic_search_weight ))
        kw_docs = self.kw_retriever.get_relevant_documents(query, int( document_limit * self.kw_search_weight ))

        return semantic_docs + kw_docs