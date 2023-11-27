import os
from flask import Flask
import pinecone
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain.document_loaders import (
    PyPDFLoader,
    OnlinePDFLoader,
    PyPDFDirectoryLoader,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone, FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.llms import HuggingFaceHub
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreRouterToolkit,
    create_vectorstore_router_agent,
)

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

        query = "Tell me five differences between cats and dogs"
        chat = ChatOpenAI(model="gpt-3.5-turbo")
        repo_id = "google/flan-t5-xxl"
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.9, "max_length": 5000}
        )
        files = [
            {
                "name": "cats",
                "path": "cats.pdf",
                "description": "use this document to answer questions about  cats",
            },
            {
                "name": "dogs",
                "path": "dogs.pdf",
                "description": "use this document to answer questions about  dogs",
            },
            {
                "name": "payment",
                "path": "payment_security_standards.pdf",
                "description": "use this document to answer questions about  payments",
            },
            {
                "name": "central_bank_of_kuwait",
                "path": "central_bank_of_kuwait.pdf",
                "description": "use this document to answer questions about  central_bank_of_kuwait",
            },
        ]

        def online_loader():
            # https://arxiv.org/pdf/2302.03803.pdf
            loader = OnlinePDFLoader(
                "https://www.flame.edu.in/pdfs/fil/presentations/FIL_Stock%20Market.pdf"
            )
            data = loader.load()
            return data

        def offline_loader(path):
            route = "example_data/" + path
            loader = PyPDFLoader(route)
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            print(docs, "Docs")
            return docs

        def directory_loader():
            loader = PyPDFDirectoryLoader("example_data/")
            docs = loader.load()
            return docs

        def create_vectorstore(file, store):
            embeddings = create_embeddings()
            vectorstore = VectorStoreInfo(
                name=file["name"], description=file["description"], vectorstore=store
            )
            return vectorstore

        def create_embeddings():
            embeddings = HuggingFaceEmbeddings()
            return embeddings

        def create_tools(file, retriever):
            tool = Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"useful when you want to answer questions about {file['name']}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            )
            return tool

        def compare_docs(question):
            vectorstores = []
            for file in files:
                docs = offline_loader(file["path"])
                print("1", file["path"])
                store = add_docs_to_pinecone(file, docs)
                print("2")
                # vectorstore = create_vectorstore(file, store)e
                vectorstore = create_tools(file, store)
                print("3")
                vectorstores.append(vectorstore)
                print("4")

            print("11")
            # router_toolkit = VectorStoreRouterToolkit(
            #     vectorstores=vectorstores, llm=llm
            # )
            # agent_executor = create_vectorstore_router_agent(
            #     llm=llm,
            #     toolkit=router_toolkit,
            #     verbose=True,
            #     handle_parsing_errors="Check your output and make sure it conforms!",
            # )
            agent_executor = initialize_agent(tools=vectorstores, llm=llm, verbose=True)
            # result = agent_executor.run(question)
            result = agent_executor({"input": f"{question}"})
            return result

        def add_docs_to_pinecone(file, docs):
            embeddings = create_embeddings()

            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV"),
            )
            index_name = os.getenv("PINECONE_INDEX_NAME")
            print(pinecone.list_indexes())
            if index_name not in pinecone.list_indexes():
                print(pinecone.list_indexes())
                pinecone.create_index(name=index_name, metric="cosine", dimension=768)

            vectorDb = Pinecone.from_documents(
                docs, embeddings, index_name=index_name, namespace=file["name"]
            )

            # vectorDb = Pinecone.from_existing_index(
            #     index_name, embeddings, namespace=file["name"]
            # )
            print(vectorDb, "vectorDb")
            return vectorDb.as_retriever()

        def query_existing_vectors(file, docs):
            embeddings = create_embeddings()
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV"),
            )
            index_name = os.getenv("PINECONE_INDEX_NAME")
            print(pinecone.list_indexes())
            docsearch = Pinecone.from_existing_index(index_name, embeddings)
            print(docsearch, "DocSearch")
            # query = "Wha is the management of Information Access"
            # result = docsearch.similarity_search(question)
            # print(result[0].page_content)

            return docsearch

        response = compare_docs(query)
        # response = query_existing_vectors(query)
        # response = add_docs_to_pinecone()
        return response
    except Exception as e:
        print("Error", e)
        return str(e)
