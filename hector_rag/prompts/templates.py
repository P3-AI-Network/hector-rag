from langchain.prompts import PromptTemplate

EXTITY_EXTRACTION_PROMPT_TEMPLATE = PromptTemplate(
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


HECTOR_QNA_PROMPT_TEMPLATE = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                You are a question answer expert!
                given below a question and knoweledge documents, read the knoweledge documents carefully 
                and answer the question. Question can have multiple subquestions so break then and answer them correctly.

                The documents may contain responses from Semantic search, Keyword Search and Graph Search so read them carefully.

                Knoweledge Documents (Context):
                {context}

                Question:
                {question}
            """
        )