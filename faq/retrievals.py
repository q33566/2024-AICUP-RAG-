import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from pathlib import Path
from openai import OpenAI

tqdm.pandas(desc="progress")


class Retrieval:
    """
    Random retrieval class. Top 1 retrieval is 0.02 acc.
    """

    def retrieval(self, question, source):
        question['retrieve'] = question['source'].apply(lambda x: np.random.choice(x))
        return question


class BM25Retrieval(Retrieval):
    def __init__(self):
        pass

    def retrieval(self, question, source):
        retrieve_list = []
        for record in question.to_dict(orient='records'):
            candidate_df = source.loc[record['source']].copy()
            bm25 = BM25Okapi(candidate_df['clean_ws'])
            candidate_df['score'] = bm25.get_scores(record['clean_ws'])
            retrieve_list.append(candidate_df.sort_values('score', ascending=False).index[0])

        question['retrieve'] = retrieve_list
        return question


class EmbeddingRetrieval(Retrieval):
    """
    Embedding retrieval class. Top 1 retrieval is 0.90 acc.
    """

    def __init__(self, api_key="",
                 model="text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_embedding(self, word_list):
        return self.client.embeddings.create(input=[''.join(word_list)], model=self.model).data[0].embedding

    def retrieval(self, question, source):
        embedding_cache = Path(self.model + '-cache')

        if not os.path.exists(embedding_cache):
            source['embedding'] = source['clean_ws'].progress_apply(lambda x: self.get_embedding(x))
            question['embedding'] = question['clean_ws'].progress_apply(lambda x: self.get_embedding(x))
            embedding_cache.mkdir(parents=True, exist_ok=True)
            question.to_csv(embedding_cache / 'question.csv')
            source.to_csv(embedding_cache / 'source.csv')
        else:
            source = pd.read_csv(embedding_cache / 'source.csv', index_col=0)
            question = pd.read_csv(embedding_cache / 'question.csv', index_col=0)
            source['embedding'] = source['embedding'].apply(lambda x: np.array(json.loads(x)))
            question['embedding'] = question['embedding'].apply(lambda x: np.array(json.loads(x)))
            question['source'] = question['source'].apply(lambda x: json.loads(x))

        retrieve_list = []
        for record in question.to_dict(orient='records'):
            candidate_df = source.loc[record['source']].copy()
            candidate_df['score'] = np.dot(np.vstack(candidate_df['embedding']), np.array(record['embedding']))
            retrieve_list.append(candidate_df.sort_values('score', ascending=False).index[0])

        question['retrieve'] = retrieve_list
        return question


class DataLoader:
    def __init__(self, question_path, answer_path, source_path):
        self.question_path = question_path
        self.answer_path = answer_path
        self.source_path = source_path

    def get_question(self):
        pass

    def get_answer(self):
        pass

    def get_source(self):
        pass


class FAQDataLoader(DataLoader):
    def __init__(self, question_path, answer_path, source_path):
        super().__init__(question_path, answer_path, source_path)

    def get_question(self):
        questions_example = json.load(open(self.question_path))['questions']
        questions_example = pd.DataFrame(questions_example).set_index('qid')
        questions_example = questions_example[questions_example['category'] == 'faq']
        questions_example['query'] = questions_example['query'].apply(lambda x: x.replace('發', '補換發'))
        return questions_example

    def get_answer(self):
        ground_truths_example = json.load(open(self.answer_path))['ground_truths']
        ground_truths_example = pd.DataFrame(ground_truths_example).set_index('qid')
        ground_truths_example = ground_truths_example[ground_truths_example['category'] == 'faq']
        return ground_truths_example

    def get_source(self):
        pid_map_content = json.load(open('../contest_dataset/contest_dataset/reference/faq/pid_map_content.json'))
        json_pattern = r'"([^"]*)"'
        pids, contents = [], []
        for pid, content in pid_map_content.items():
            content = ''.join(
                match for match in re.findall(json_pattern, json.dumps(content, ensure_ascii=False))
                if match not in ['question', 'answers'])
            contents.append(content)
            pids.append(int(pid))
        return pd.DataFrame({'pid': pids, 'content': contents}).set_index('pid')


class DataPreprocessor:
    """
    model name = [bert-base, albert-base, bert-tiny, albert-tiny]
    * bert-base + BM25: 1 acc.
    * albert-base + BM25: 0.94 acc.
    * bert-tiny + BM25: 0.90 acc.
    * albert-tiny + BM25: 0.94 acc.

    * bert-tiny + Embedding: 0.90 acc.
    """

    def __init__(self, model="bert-base"):
        self.ws_driver = CkipWordSegmenter(model=model, device=0)
        self.pos_driver = CkipPosTagger(model=model, device=0)
        self.ner_driver = CkipNerChunker(model=model, device=0)

    def preprocess(self, series):
        data_frame = series.to_frame()

        data_frame['content_ws'] = self.ws_driver(data_frame[series.name])
        data_frame['content_pos'] = self.pos_driver(data_frame['content_ws'])
        data_frame['content_ner'] = [[ner.word for ner in ner_list if len(ner.word) > 1] for ner_list in
                                     self.ner_driver(series)]

        clean_ws_list = []
        for content_ws, content_pos, content_ner in zip(data_frame['content_ws'], data_frame['content_pos'],
                                                        data_frame['content_ner']):
            clean_ws = []
            for ws, pos in zip(content_ws, content_pos):
                if pos.startswith('V') or pos.startswith('N') or ws in content_ner:
                    if len(ws) > 1:
                        clean_ws.append(ws)
            clean_ws_list.append(clean_ws)
        data_frame['clean_ws'] = clean_ws_list

        return data_frame['clean_ws']
