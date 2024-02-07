import os
from langchain.chat_models import ChatOpenAI
from pathlib import Path
import pinecone
from langchain.document_loaders import (
    PyPDFLoader,
    OnlinePDFLoader,
    PyPDFDirectoryLoader,
)
import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone, FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from dotenv import load_dotenv
from pydantic.v1 import BaseModel, Field
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.llms import HuggingFaceHub
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreRouterToolkit,
    create_vectorstore_router_agent,
)

# export OPENAI_API_KEY="..."
load_dotenv()


def add_documents():
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

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
                "name": "MAS",
                "path": "MAS.pdf",
                "description": "use this document to answer questions about technology risk guidelines",
            },
            {
                "name": "SAMA",
                "path": "SAMA.pdf",
                "description": "use this document to answer questions about  saudi arabia monetary authority",
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

        def directory_loader():
            loader = PyPDFDirectoryLoader("example_data/")
            docs = loader.load()
            return docs

        def online_loader():
            # https://arxiv.org/pdf/2302.03803.pdf
            loader = OnlinePDFLoader(
                "https://www.flame.edu.in/pdfs/fil/presentations/FIL_Stock%20Market.pdf"
            )
            pages = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            return docs

        def offline_loader(path):
            route = "example_data/" + path
            loader = PyPDFLoader(route)
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            print(docs, "Docs")
            return docs

        def create_embeddings():
            # embeddings = HuggingFaceEmbeddings()
            embeddings = OpenAIEmbeddings()
            return embeddings

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

            vectorDb = Pinecone.from_documents(docs, embeddings, index_name=index_name)

            print(vectorDb, "vectorDb")
            return vectorDb.as_retriever()

        for file in files:
            docs = offline_loader(file["path"])
            add_docs_to_pinecone(file, docs)
            print(file["path"])

        return "Docuemnt added successfully"
    except Exception as e:
        raise e


def home(query):
    try:

        class DocumentInput(BaseModel):
            question: str = Field()

        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        repo_id = "google/flan-t5-xxl"
        # llm = HuggingFaceHub(
        #     repo_id=repo_id, model_kwargs={"temperature": 0.9, "max_length": 5000}
        # )
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
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
                "name": "MAS",
                "path": "MAS.pdf",
                "description": "use this document to answer questions about technology risk guidelines",
            },
            {
                "name": "SAMA",
                "path": "SAMA.pdf",
                "description": "use this document to answer questions about  saudi arabia monetary authority",
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

        def create_vectorstore(file, store):
            embeddings = create_embeddings()
            vectorstore = VectorStoreInfo(
                name=file["name"], description=file["description"], vectorstore=store
            )
            return vectorstore

        def create_embeddings():
            # embeddings = HuggingFaceEmbeddings()
            embeddings = OpenAIEmbeddings()
            return embeddings

        def create_tools(file, retriever):
            tool = Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"useful when you want to answer questions about {file['name']}",
                func=RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                ),
            )
            return tool

        def compare_docs(question):
            vectorstores = []
            for file in files:
                store = add_docs_to_pinecone()
                vectorstore = create_tools(file, store)
                vectorstores.append(vectorstore)

            agent_executor = initialize_agent(
                tools=vectorstores,
                llm=llm,
                verbose=True,
                agent=AgentType.OPENAI_FUNCTIONS,
            )

            result = agent_executor({"input": f"{question}"})
            return result

        def add_docs_to_pinecone():
            embeddings = create_embeddings()

            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV"),
            )
            index_name = os.getenv("PINECONE_INDEX_NAME")

            vectorDb = Pinecone.from_existing_index(
                index_name,
                embeddings,
            )
            print(vectorDb, "vectorDb")
            return vectorDb.as_retriever()

        response = compare_docs(query)
        # response = query_existing_vectors(query)
        # response = add_docs_to_pinecone()
        return response
    except Exception as e:
        print("Error", e)
        return str(e)

def main():
    st.title('Compare documents')
    query = st.text_input("Ask to comapre documents")

    if st.button("Add Docs"):
        add_documents()
    if query:
        query_result = home(query)
        st.write(query_result)

if __name__ == "__main__":
    main()