import os
import warnings
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from itertools import count
from dotenv import load_dotenv


load_dotenv()
warnings.filterwarnings("ignore", message="Multiple definitions in dictionary")


class Embedder:
    def __init__(self, root_path):
        self.root_path = Path(root_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
        )

        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            api_key=os.getenv('API_KEY'),
            dimensions=1024
        )

        self.vector_store = Chroma(
            collection_name="finance",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )

    def read_insurance(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for page in pages:
            page.metadata['source'] = file_path.stem
            page.metadata['category'] = file_path.parent.name
        docs = self.text_splitter.split_documents(pages)
        return docs

    def read_faq(self, file_path):
        json_data = json.load(open(file_path))

        docs = []
        for source, qa_pairs in json_data.items():
            c = count(0)
            for qa_pair in qa_pairs:
                for answer in qa_pair['answers']:
                    docs.append(
                        Document(
                            page_content=f"{qa_pair['question']}\n{answer}",
                            metadata={'source': source, 'category': 'faq', 'page': next(c)},
                        )
                    )
        return docs

    def read_finance(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for page in pages:
            page.metadata['source'] = file_path.stem
            page.metadata['category'] = file_path.parent.name
        docs = self.text_splitter.split_documents(pages)
        return docs

    def embed(self):
        files_path = list(self.root_path.rglob('*.*'))

        docs = []
        for file_path in files_path:
            if file_path.parent.name == 'insurance':
                docs.extend(self.read_insurance(file_path))
            elif file_path.parent.name == 'faq':
                docs.extend(self.read_faq(file_path))
            elif file_path.parent.name == 'finance':
                docs.extend(self.read_finance(file_path))

        self.vector_store.add_documents(docs, ids=[str(uuid4()) for _ in range(len(docs))])

if __name__ == '__main__':
    embedder = Embedder('/home/xunhaoz/PycharmProjects/RAGAndLLMInFinance/contest_dataset/contest_dataset/reference')
    embedder.embed()
