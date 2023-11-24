from flask import Flask
import os
import getpass
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

app = Flask(__name__)

# export OPENAI_API_KEY="..."
load_dotenv()

@app.route("/")
def home():
    try:
        # os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
        print(os.environ.get('OPENAI_API_KEY'))
        chat = ChatOpenAI(model="gpt-3.5-turbo",)
        def online_loader():
            # https://arxiv.org/pdf/2302.03803.pdf
            loader = OnlinePDFLoader("https://www.flame.edu.in/pdfs/fil/presentations/FIL_Stock%20Market.pdf") 
            data = loader.load()
            return data
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(online_loader())
        db = Chroma.from_documents(texts, HuggingFaceEmbeddings())
        embeddings = HuggingFaceEmbeddings().embed_query("")
        print("Embeddings:",embeddings)
        docs = db.similarity_search_by_vector(embeddings)
        print("Docs",docs[0].page_content)
        # retriever = db.as_retriever()
        # print("Retriever:",retriever)
        # docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
        # print("Docs:")
        return(str(docs[0].page_content))
    except Exception as e:
        print("Error",e)
        return(str(e)) 
