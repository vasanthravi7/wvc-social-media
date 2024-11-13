import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import chromadb
load_dotenv()
api_type = 2 # 1 - Azure OpenAI, 2- OpenAI

if api_type == 1:
    # Load environment variables
    azure_endpoint = os.getenv("Azure_API_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-03-15-preview"
    deployment_name = "gpt-4o"  # Ensure this matches your deployment name in Azure

    # Initialize embeddings and model
    embeddings = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint,api_key=api_key)
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        model=deployment_name,
        temperature=1,
        max_tokens=300,
    )
else:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    llm = OpenAI()

# Define paths and parameters
persistent_directory = "./chroma_db/World-Vision-Canada-FY23-Annual-Results-Report"
# Step 1: Load the existing Chroma DB instance
chroma_db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# client = chromadb.PersistentClient(path=persistent_directory)  # or HttpClient()
# collections = client.list_collections()

# # Print the number of collections
# for collection in collections:
#     print(collection.name)
# openai_api = OpenAI()

# Define a simple query
query = "Letter from president"

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