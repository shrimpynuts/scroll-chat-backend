from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
import os
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import pickle
import time
from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)


class Page:
    def __init__(self, url, text):
        self.url = url
        self.text = text

    def __str__(self):
        return f"{self.url}:\n{self.text}\n)"


def fetchPages(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    pages = []
    for link in soup.find_all('a'):
        if link.get('href').startswith('/'):
            pages.append(link.get('href'))
    return [url + page for page in pages]


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def fetchPageContent(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


def fetchAllPages(url):
    pages = fetchPages(url)
    return [Page(page, fetchPageContent(page)) for page in pages]


def generate_search_index(source_docs):
    source_chunks = []
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in source_docs:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(
                Document(page_content=chunk, metadata=source.metadata))

    search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
    return search_index


def store_search_index(search_index, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(search_index, f)


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


app = Flask(__name__)
file = open('search_index_new.pickle', 'rb')
source_index = pickle.load(file)
file.close()

# pages = fetchAllPages('https://guide.scroll.io')
# source_docs = [Document(page_content=page.text, metadata={
#                         "source": page.url}) for page in pages]
# source_index = generate_search_index(source_docs)


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
