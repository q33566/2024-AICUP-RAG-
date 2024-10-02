import os
import random

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


class Retriver:
    def __init__(self, database):
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            api_key=os.getenv('API_KEY'),
            dimensions=1024
        )

        self.vector_store = Chroma(
            collection_name="finance",
            embedding_function=self.embeddings,
            persist_directory=database,
        )

    def retrivel(self, qid, source, query, category):
        source, source_score = list(map(str, source)), [0 for _ in range(len(source))]
        where = {
            "$and": [
                {"source": {"$in": source}},
                {"category": {"$in": [category]}}
            ]
        }
        retriever = self.vector_store.as_retriever(search_kwargs={"filter": where, "k": 40})
        results = retriever.invoke(query)

        for result in results:
            if result.metadata['source'] in source:
                source_score[source.index(result.metadata['source'])] += 1

        max_search_source = source[source_score.index(max(source_score))]
        return {'qid': qid, 'retrieve': int(max_search_source), 'category': category}


class RetriverWithChat(Retriver):
    def __init__(self, database):
        super().__init__(database=database)
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            api_key=os.getenv('API_KEY')
        )

    def completion(self, query):
        return self.llm.invoke(query).content

    def retrivel(self, qid, source, query, category):
        query = self.completion(query)
        source, source_score = list(map(str, source)), [0 for _ in range(len(source))]
        where = {
            "$and": [
                {"source": {"$in": source}},
                {"category": {"$in": [category]}}
            ]
        }
        retriever = self.vector_store.as_retriever(search_kwargs={"filter": where, "k": 40})
        results = retriever.invoke(query)

        for result in results:
            if result.metadata['source'] in source:
                source_score[source.index(result.metadata['source'])] += 1

        max_search_source = source[source_score.index(max(source_score))]
        return {'qid': qid, 'retrieve': int(max_search_source), 'category': category}


class RandomRetriver(Retriver):
    def __init__(self):
        pass

    def retrivel(self, qid, source, query, category):
        random_choice = random.choice(source)
        return {'qid': qid, 'retrieve': random_choice, 'category': category}
