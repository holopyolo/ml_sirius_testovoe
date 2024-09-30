import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List
import re
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import Document
import os
import sys
from sentence_transformers import SentenceTransformer

# семантик и иерарх не меняют скор
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    HierarchicalNodeParser
)


# кастом клинер для преобработки контекста
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs) -> list[BaseNode]:
        for node in nodes:
            node.text = node.text.replace('\t', ' ')
            node.text = node.text.replace('\n', ' ')
        return nodes


CHUNK_SIZE = 250
CHUNK_OVERLAP = 70
text_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),
        text_splitter,
    ]
)


# получаем llm и tokenizer
def get_llm(llm_hf_name,
            model_args={}):
    tokenizer = AutoTokenizer.from_pretrained(llm_hf_name)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_hf_name, **model_args).eval()
    generation_config = GenerationConfig.from_pretrained(llm_hf_name)
    return (llm, tokenizer, generation_config)

# получаем модель энкодер


def get_embedder(model_name):
    embedder = SentenceTransformer(model_name)

    def f(texts, **args):
        return embedder.encode(texts, normalize_embeddings=True, **args)
    return f


def parse_questions(output):
    output = output.lower()
    question_pattern = r'вопрос \d+ *:.+'
    questions = re.findall(question_pattern, output)
    questions = [q[q.find(':')+1:].strip() for q in questions]
    return questions


def create_queries(llm, tokenizer, generation_config, question):
    prompt_desired = f'''На основе вопроса: {question} создай 2-3 абсолютно вопроса схожих по смыслу, которые могли бы быть заданы в том же контексте но переписаны другими словами\n
                         Формат: Вопрос 1: ...\n Вопрос 2: ...\n Вопрос 3: ...'''
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": prompt_desired
    }], tokenize=False, add_generation_prompt=True)
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(llm.device) for k, v in data.items()}
    output_ids = llm.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
   # print(output)
    print('letsooo go')
    try:
        questions_cands = parse_questions(output)
        questions_cands[-1] = question  # изначальный вопрос оставляем.
    except (Exception) as e:
        print(e)
        return [question]
    print('Вопросы сгенерированы')
    print(questions_cands)
    return questions_cands

# ranked_lists: словарь, где для каждого вопроса мы получили релевантные документы по убыванию


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    print(ranked_lists)
    for rank_list in ranked_lists:
        for rank, item in enumerate(rank_list):
            if item not in scores:
                scores[item] = 0
            scores[item] += 1 / (rank + 1 + k)
    print('Ранжирование пройдено')
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# k: кол-во частей, которые обработает каждый retriever, для дальнейшего reranking
# k_final: top-k итоговых частей текста, которые пойдут ллм


def hybrid_retriever(embedder, texts, retriever, question, llm, tokenizer, generation_config, k=1, k_final=5):
    qustionds_cands = create_queries(
        llm, tokenizer, generation_config, question)
    ranked_lists = []
    for candidat_q in qustionds_cands:
        if '?' not in candidat_q:
            continue
        ranked_lists.append(
            retriever(embedder, texts, candidat_q, k, get_indices=True))

    scores = reciprocal_rank_fusion(ranked_lists)[:k_final]  # (index, score)
    return [texts[ind] for ind, score in scores]


def prepare_context(context, pipeline):
    docs = [Document(text=context)]
    nodes = pipeline.run(documents=docs)
    return [n.text for n in nodes]
#


def retrieve_context_topk(embedder, texts, question, k=1, get_indices=False):
    embeddings = embedder(texts)
    question_embedding = embedder([question])[0]
    scores = np.dot(embeddings, question_embedding) / \
        (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding))
    top_k_indices = np.argsort(scores)[-k:]
    if get_indices:
        return [i for i in reversed(top_k_indices)]
    return [texts[i] for i in reversed(top_k_indices)]


def pipeline_rag_answer(llm,
                        tokenizer,
                        question,
                        context,
                        embedder_model,
                        generation_config,
                        pipeline,
                        prompt,
                        retriever_pipeline,
                        k):
    relevant_content = ''.join(retriever_pipeline(embedder=embedder_model,
                                                  texts=prepare_context(
                                                      context, pipeline),
                                                  retriever=retrieve_context_topk,
                                                  question=question,
                                                  llm=llm,
                                                  k=k,
                                                  tokenizer=tokenizer,
                                                  generation_config=generation_config))
    relevant_content = relevant_content.replace('\n', ' ')
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": ""
    }, {
        "role": "user",
        "content": prompt.format(context_str=relevant_content, query_str=question)
    }], tokenize=False, add_generation_prompt=True)
    print('Начало генерации ответа')
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(llm.device) for k, v in data.items()}
    output_ids = llm.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]  # урзаем нач. промпт
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return output


def prepare(row):
    for k, v in row.items():
        if isinstance(v, str):
            row[k] = v.lower()
        elif isinstance(v, list):
            row[k] = [x.lower() for x in v]
    row['answers']['text'] = [row['answers']['text'][0].lower()]
    return row
