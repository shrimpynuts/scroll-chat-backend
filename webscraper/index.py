from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from bs4.element import Comment


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


customPages = [
    "https://guide.scroll.io/",
    "https://scrollzkp.notion.site/Scroll-Brand-Assets-PUBLIC-8522d7dbe4c745579d3e3b14f3bbecc0",
    "https://github.com/scroll-tech",
    "https://guide.scroll.io/user-guide/common-errors#incorrect-nonce-error-when-sending-a-transaction-in-metamask",
    "https://scroll.io/",
    "https://scroll.io/",
    "https://scroll.io/blog",
    "https://scroll.io/team",
    "https://scroll.io/join-us",
    "https://twitter.com/Scroll_ZKP",
    "https://www.youtube.com/@Scroll_ZKP",
    "https://scroll.io/prealpha/",
    "https://scroll.io/blog/visionAndValues",
    "https://scroll.io/blog/zkEVM",
    "https://scroll.io/blog/preAlphaTestnet",
    "https://scroll.io/blog/technicalPrinciples",
    "https://scroll.io/blog/architecture",
    "https://scroll.io/blog/upgradingPreAlphaTestnet",
    "https://scroll.io/blog/kzg",
    "https://scroll.io/blog/proofGeneration",
    "https://scroll.io/blog/releaseNote0109",
]

customPagesFormatted = [Page(page, fetchPageContent(page))
                        for page in customPages]


# pages = fetchAllPages('https://guide.scroll.io')
# source_docs = [Document(page_content=page.text, metadata={
#                         "source": page.url}) for page in pages]
# source_index = generate_search_index(source_docs)
