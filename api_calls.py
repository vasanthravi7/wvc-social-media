import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from crewai_tools import tool
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from crewai import LLM
app = Flask(__name__)

# Load environment variables
load_dotenv()

# API Type Configuration
api_type = 2  # 1 - Azure OpenAI, 2 - OpenAI

if api_type == 1:
    azure_endpoint = os.getenv("Azure_API_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-03-15-preview"
    deployment_name = "gpt-4o"
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
    llm = OpenAI(max_tokens=300)

# Define paths and parameters
persistent_directory = "./chroma_db/World-Vision-Canada-FY23-Annual-Results-Report"
chroma_db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Custom tool for querying Chroma DB
@tool
def query_chroma_db(query_text: str) -> str:
    """
    Retrieves relevant chunks from Chroma DB using a semantic search
    based on the input query_text.

    Args:
        query_text (str): The text to query Chroma DB with.

    Returns:
        str: Formatted results from the Chroma DB, containing relevant chunks.
    """
    # Perform a similarity search in Chroma DB
    results = chroma_db.similarity_search(query_text, k=10)  # Retrieve top 10 relevant chunks
    
    # Extract 'page_content' from each Document in results
    formatted_results = "\n\n".join([doc.page_content for doc in results if hasattr(doc, 'page_content')])
    
    return formatted_results

# Data Retrieval Agent
data_retriever = Agent(
    role="Data Researcher",
    goal="Retrieve relevant information about the blog topic from the Chroma vector database.",
    verbose=True,
    memory=True,
    backstory="An AI specialized in locating relevant research and information from a vast Chroma DB.",
    tools=[query_chroma_db]
)

# Blog Writing Agent
blog_writer = Agent(
    role="Blog Writer",
    goal="Write an informative blog post using the retrieved data from Chroma DB and referencing sample_blog.txt.",
    verbose=True,
    memory=True,
    backstory="An AI writer proficient in crafting insightful blog posts that clarify complex topics."
)

# Instagram Conversion Agent
instagram_converter = Agent(
    role="Instagram Post Creator",
    goal=(
        "Convert the blog post to an engaging Instagram post. "
        "Make it short, visually appealing, and use emojis and hashtags appropriately. "
        "Ensure it aligns with the style in sample_instagram.txt."
    ),
    verbose=True,
    memory=True,
    backstory="A creative AI specializing in summarizing blogs into visually appealing Instagram posts."
)

# LinkedIn Conversion Agent
linkedin_converter = Agent(
    role="LinkedIn Post Creator",
    goal=(
        "Convert the blog post to a professional LinkedIn post. "
        "Focus on detailed insights and a polished tone with minimal use of emojis. "
        "Ensure it aligns with the style in sample_linkedin.txt."
    ),
    verbose=True,
    memory=True,
    backstory="A professional AI that distills blog content into insightful LinkedIn posts."
)

# Define tasks
retrieval_task = Task(
    description="Retrieve information relevant to {blog_topic} from Chroma DB.",
    expected_output="Summarized insights related to {blog_topic}.",
    agent=data_retriever
)

writing_task = Task(
    description="Write a comprehensive blog post on {blog_topic}.",
    expected_output="A Markdown blog post on {blog_topic}.",
    agent=blog_writer,
    async_execution=False,
    output_file="outputs/blog_chroma_collection_name.txt"
)

instagram_task = Task(
    description="Convert the blog post to an Instagram-friendly format.",
    expected_output="Instagram post text based on the blog content.",
    agent=instagram_converter,
    async_execution=False,
    output_file="outputs/instagram_post_chroma_collection_name.txt"
)

linkedin_task = Task(
    description="Convert the blog post to a LinkedIn-friendly format.",
    expected_output="LinkedIn post text based on the blog content.",
    agent=linkedin_converter,
    async_execution=False,
    output_file="outputs/linkedin_post_chroma_collection_name.txt"
)

# Assemble Crew with a sequential process
crew = Crew(
    agents=[data_retriever, blog_writer, instagram_converter, linkedin_converter],
    tasks=[retrieval_task, writing_task, instagram_task, linkedin_task],
    process=Process.sequential
)

# Function to dynamically generate file path
def generate_output_file_path(collection_name: str, file_type: str) -> str:
    return os.path.join("outputs", f"{file_type}_{collection_name}.txt")

output_directory = 'outputs'
@app.route('/generate_output', methods=['POST'])
def generate_output():
    try:
        # Extract 'blog_topic' from the request JSON
        data = request.get_json()
        blog_topic = data.get('blog_topic')

        if not blog_topic:
            return jsonify({"error": "Missing 'blog_topic' parameter"}), 400

        # Determine collection name (assuming persistent_directory is defined somewhere)
        collection_name = persistent_directory.split('/')[-1]   
        blog_output_file = writing_task.output_file = generate_output_file_path(collection_name, "blog")
        instagram_output_file = instagram_task.output_file = generate_output_file_path(collection_name, "instagram_post")
        linkedin_output_file =linkedin_task.output_file = generate_output_file_path(collection_name, "linkedin_post")    
        # Generate output file paths for blog, instagram post, and linkedin post
        
        # # Execute the crew process
        result = crew.kickoff(inputs={'blog_topic': blog_topic})
        # # Check if the result is successful and files are created
        if not all(os.path.exists(file) for file in [blog_output_file, instagram_output_file, linkedin_output_file]):
            return jsonify({"error": "One or more output files not generated"}), 500
        
        # Read the contents of the output files
        with open(blog_output_file, 'r', encoding='utf-8') as file:
            blog_content = file.read()
        
        with open(instagram_output_file, 'r', encoding='utf-8') as file:
            instagram_content = file.read()
        
        with open(linkedin_output_file, 'r', encoding='utf-8') as file:
            linkedin_content = file.read()

        # Return the contents as a JSON response
        return jsonify({
            "blog_content": blog_content,
            "instagram_content": instagram_content,
            "linkedin_content": linkedin_content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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


if __name__ == '__main__':
    app.run(debug=True)