import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from flask import Flask, request, jsonify

# Load environment variables from .env file
load_dotenv()

# API configuration (1 - Azure OpenAI, 2 - OpenAI)
api_type = 2  # 1 - Azure OpenAI, 2- OpenAI

# Initialize embeddings and language model based on API type
if api_type == 1:
    azure_endpoint = os.getenv("Azure_API_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-03-15-preview"
    deployment_name = "gpt-4o"  # Ensure this matches your deployment name in Azure

    # Initialize embeddings and model for Azure OpenAI
    embeddings = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint, api_key=api_key)
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

# Load the existing Chroma DB instance
chroma_db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Initialize Flask app
app = Flask(__name__)

# Function to handle queries and return chatbot responses
def get_chatbot_response(query):
    # Step 1: Perform similarity search in Chroma DB
    results = chroma_db.similarity_search(query, k=5)  # Get top 5 most similar results
    # print(results)
    # Step 2: Extract relevant information from the results
    # Combine the content of the results for better context in the prompt
    context = "\n".join([result.page_content for result in results])
    
    # Step 3: Construct a proper prompt template for LLM
    prompt_template = """
    Use the following information to answer the user's question:
    
    {context}
    
    User's Question: {query}
    
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)
    full_prompt = prompt.format(context=context, query=query)

    # Step 4: Pass the formatted prompt to the LLM
    response = llm.invoke(full_prompt)
    return response

@app.route("/chat", methods=["POST"])
def chat():
    # Get user query from the POST request
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query parameter is required."}), 400

    # Get the response from the chatbot
    try:
        response = get_chatbot_response(query)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)