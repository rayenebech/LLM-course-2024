from pydantic import BaseModel, Field
from IPython.display import HTML
from dotenv import load_dotenv
from typing import List
from tqdm import tqdm
import logging
import sys
import glob
import os

from langtrace_python_sdk import (
    langtrace,
) 
import qdrant_client


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llmsherpa.readers import LayoutPDFReader
from llama_index.llms.gemini import Gemini


from helpers import load_config, load_files

load_dotenv()

langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


PDF_URL = "https://abc.xyz/assets/91/b3/3f9213d14ce3ae27e1038e01a0e0/2024q1-alphabet-earnings-release-pdf.pdf"


class LlamaIndexManager:
    def __init__(self, collection_name="pdf_collection"):
        self.config = load_config("config.yaml")
        self.collection_name = collection_name
        Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.__initialize_llm()
        self.qdrant_client = qdrant_client.QdrantClient(url=self.config["rag"]["qdrant_url"])

    def __initialize_llm(self):
        self.llm = Gemini(model= self.config["rag"]["responder"])

    def __initialize_rag(self):
        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self.query_engine = self.index.as_query_engine(
            llm=self.llm, 
            similarity_top_k=4,
            verbose=True, 
            streaming=False
            )
    
    def read_pdf(self, pdf_url):
        pdf_reader = LayoutPDFReader(self.config["parsing"]["llmsherpa_api_url"])
        return pdf_reader.read_pdf(pdf_url)

    def create_index(self, doc, file_name):
        vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            [Document(text=chunk.to_context_text(), metadata={"filename": file_name}) for chunk in doc.chunks()],
            storage_context=storage_context,
        )
        
    def index_all_docs(self, root_dir):
        files = load_files(root_dir)
        for file in tqdm(files):
            doc = self.read_pdf(file)
            file_name = os.path.basename(file)
            print(f"Indexing {file_name}...")
            self.create_index(doc, file_name)

    def index_doc(self, file: str):
        doc = self.read_pdf(file)
        if file.startswith("http"):
            file_name = file.split("/")[-1]
        else:
            file_name = os.path.basename(file)
        print(f"Indexing {file_name}...")
        self.create_index(doc, file_name)

    def query(self, query_text):
        self.__initialize_rag()
        response = self.query_engine.query(query_text)
        return response
    
if __name__ == "__main__":
    manager = LlamaIndexManager()
    # Please first index the documents before querying using the following line
    # manager.index_doc(PDF_URL)

    # query  = "What % Net income is of the Revenues?"
    # query = "What was Google's operating margin for 2024"
    query = "How much is the total operating income for Google Services and Google Cloud?"

    response = manager.query(query)
    print(response)