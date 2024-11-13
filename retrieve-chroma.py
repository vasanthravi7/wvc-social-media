import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
load_dotenv()
# openai_api = OpenAI()
azure_endpoint = os.getenv("OPENAI_API_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
api_version = "2023-03-15-preview"
deployment_name = "gpt-4o"  # Make sure this matches your deployment name in Azure
# Initialize OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint)

# Define paths and parameters
persistent_directory = "./chroma_db"
# Step 1: Load the existing Chroma DB instance
chroma_db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)
# openai_api = OpenAI()

# Define a simple query
query = "president letter"

# Perform a similarity search in Chroma DB
# The query will be embedded and compared with the stored embeddings
results = chroma_db.similarity_search(query, k=10)  # Get top 3 most similar results

# Print out the results with metadata
for result in results:
    print(result)
    print("*******************")
    # print(f"Topic: {result.metadata['topic']}")
    # print(f"Page Range: {result.metadata['start_page']} - {result.metadata['end_page']}")
    # print(f"Content Chunk: {result.page_content}")