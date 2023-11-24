from flask import Flask
import os
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.llms import HuggingFaceHub

app = Flask(__name__)

# export OPENAI_API_KEY="..."
load_dotenv()


@app.route("/")
def home():
    try:
        class DocumentInput(BaseModel):
            question: str = Field()
        # Create a temporary directory in the script's folder
        # script_dir = Path(__file__).resolve().parent
        # temp_dir = os.path.join(script_dir, "tempDir")

        query = "Summarize the key points of technology_risk_management document"
        chat = ChatOpenAI(model="gpt-3.5-turbo")
        repo_id = "google/flan-t5-xxl"
        llm = HuggingFaceHub( repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 2024})
        files = [
                {
                    "name": "technology_risk_management",
                    "path": "technology_risk_management.pdf"
                },
                # {
                #     "name": "b.pdf",
                #     "path": "b.pdf"
                # }
            ]
        
        def online_loader():
            # https://arxiv.org/pdf/2302.03803.pdf
            loader = OnlinePDFLoader("https://www.flame.edu.in/pdfs/fil/presentations/FIL_Stock%20Market.pdf") 
            data = loader.load()
            return data
        
        def offline_loader(path):
            route = "example_data/" + path
            loader = PyPDFLoader(route)
            pages = loader.load_and_split()
            print(pages)
            print(route)
            return pages

        def directory_loader():
            loader = PyPDFDirectoryLoader("example_data/")
            docs = loader.load_and_split()
            return docs
        
        def create_chain(docs):
            embeddings = create_embeddings()
            pdf_search = Chroma.from_documents(docs, embeddings)
            chain = ConversationalRetrievalChain.from_llm(llm, retriever=pdf_search.as_retriever(search_kwargs={"k": 1}, return_source_documents=True))
            return chain
        
        def create_embeddings():
            embeddings = HuggingFaceEmbeddings()
            return embeddings
            
        def retriever(docs):
            embeddings = create_embeddings()
            retriever = FAISS.from_documents(docs, embeddings).as_retriever()
            return retriever
    
        def compare_docs(question):
            tools = []
            # parsed_documents = directory_loader()
            for file in files:
                pages = offline_loader(file["path"])
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                documents = text_splitter.split_documents(pages)
                retrieved_docs = retriever(documents)
                tools.append(Tool(args_schema=DocumentInput,name=file["name"], description=f"useful when you want to answer questions about {file['name']}", func=RetrievalQA.from_chain_type(llm=llm, retriever=retrieved_docs)))
            
            agent = initialize_agent(tools=tools, llm=llm, verbose=True ,handle_parsing_errors=True)

            result = agent({"input": question})
            print(result)
            return result
        
        def query_single_doc(question):
            pages = offline_loader("technology_risk_management.pdf")
            # This text splitter is used to create the child documents
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(
                collection_name="full_documents", embedding_function=HuggingFaceEmbeddings()
            )
            
            # The storage layer for the parent documents
            store = InMemoryStore()
            retriever = ParentDocumentRetriever(
                vectorstore= vectorstore,
                docstore=store,
                child_splitter=child_splitter,
            )
            
            retriever.add_documents(pages, ids=None)
            list(store.yield_keys())
            sub_docs = vectorstore.similarity_search("technology risk management")
            print(sub_docs[0].page_content)
            return str(sub_docs[0].page_content)
        
        # response = compare_docs(query)
        response = query_single_doc(query)
        return response
    except Exception as e:
        print("Error",e)
        return(str(e)) 
