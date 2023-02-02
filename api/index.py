from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
import pickle
import time
from flask import Flask, jsonify, request


pickle_filename = 'search_index_website_and_twitter.pickle'

st = time.time()
app = Flask(__name__)
file = open(pickle_filename, 'rb')
source_index = pickle.load(file)
file.close()
print(time.time() - st)


def generate_answer(question, index, chain):
    return (
        chain(
            {
                "input_documents": index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )


@app.route('/')
def home():
    return "Hello, World!"


@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    st = time.time()
    chain = load_qa_with_sources_chain(OpenAI(temperature=0))
    data = request.get_json()
    question = data['question']
    answer = generate_answer(question, source_index, chain)
    et = time.time()
    time_elapsed = et - st
    return jsonify(
        question=question,
        answer=answer,
        time_elapsed=time_elapsed
    )
