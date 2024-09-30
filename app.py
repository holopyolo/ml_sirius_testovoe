from flask import Flask, request, jsonify
import argparse
import torch
from utils import *

RAG_PROMPT = (
    "Контекстная информация снизу. (набор документов разделенных переносом строки)\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Используя только данную информацию контекст, "
    "Ответь на вопрос. Ты знаешь что ответ явно содержится в контексте. Дай прямой ответ ОЧЕНЬ короткий ответ\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)

based_args_model = {
        "load_in_8bit": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2"}

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Run a Flask application.')
parser.add_argument('--llm', type=str, default='IlyaGusev/saiga_llama3_8b', help='поменять модель')
parser.add_argument('--embedder', type=str, default='deepvk/USER-bge-m3', help='поменять модель')
parser.add_argument('--flashdown', type=int, default=0, help='Отключить flash attention, если есть проблемы с установкой')
args = parser.parse_args()

embedder_model = get_embedder(args.embedder)
llm, tokenizer, generation_config = None, None, None
if args.flashdown:
    llm, tokenizer, generation_config = get_llm(llm_hf_name=args.llm,
                                      model_args=based_args_model.pop("attn_implementation"))
else:
    llm, tokenizer, generation_config = get_llm(llm_hf_name=args.llm,
                                          model_args=based_args_model)


@app.route('/submit', methods=['POST'])
def submit_strings():
    data = request.get_json()

    context = data['context']
    question = data['question']
    
    #такие настройки обеспечивают оптимальный скор
    Answer = pipeline_rag_answer(
       llm=llm, 
       tokenizer=tokenizer,
       question=question,
       embedder_model=embedder_model,
       context=context,
       generation_config=generation_config,
       pipeline=pipeline,
       prompt=RAG_PROMPT,  
       retriever_pipeline=hybrid_retriever,
       k=15
    )
    
    
    return jsonify({
        'Answer': Answer
    }), 200



if __name__ == '__main__':
    app.run(debug=False)
