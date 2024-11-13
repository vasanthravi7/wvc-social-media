import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from crewai_tools import tool
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from crewai import LLM

# Load environment variables
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
    llm = OpenAI(max_tokens=300)

# Define paths and parameters
persistent_directory = "./chroma_db/World-Vision-Canada-FY23-Annual-Results-Report"

# Step 1: Load the existing Chroma DB instance
chroma_db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Custom tool for querying Chroma DB
@tool
def query_chroma_db(query_text: str) -> str:
    """
    This function retrieves relevant chunks from Chroma DB using a semantic search
    based on the input query_text.
    """
    # Perform a similarity search in Chroma DB
    results = chroma_db.similarity_search(query_text, k=10)  # Retrieve top 10 relevant chunks
    
    # Extract 'page_content' from each Document in results
    if isinstance(results, list):
        formatted_results = "\n\n".join([doc.page_content for doc in results if hasattr(doc, 'page_content')])
    else:
        formatted_results = "No results found or invalid format."
    print("here")
    print(formatted_results)
    return formatted_results

# Agent to retrieve relevant data from Chroma DB
data_retriever = Agent(
    role="Data Researcher",
    goal="Retrieve relevant information about the blog topic from the Chroma vector database.",
    verbose=True,
    memory=True,
    backstory="You're adept at quickly locating relevant content and insights.",
    tools=[query_chroma_db]
)

# Agent to write the blog post based on retrieved data and reference sample_blog.txt
blog_writer = Agent(
    role="Blog Writer",
    goal="Write an informative blog post using the retrieved data from Chroma DB and referencing sample_blog.txt.",
    verbose=True,
    memory=True,
    backstory="You craft insightful and engaging articles, making complex information easy to understand.",
    # llm = llm
)

# Define the retrieval task for data collection
retrieval_task = Task(
    description=(
        "Search the Chroma database for information relevant to {blog_topic}. "
        "Collect and organize insights to support the creation of a blog post."
    ),
    expected_output="A collection of summarized insights and examples related to {blog_topic}.",
    agent=data_retriever
)

# Define the writing task for blog generation
writing_task = Task(
    description=(
        "Use the gathered information to write a comprehensive blog post on {blog_topic}. "
        "Structure the blog to include an introduction, main insights, and a conclusion. "
        "Use 'sample_blog.txt' for reference."
    ),
    expected_output="A blog post in Markdown format covering the topic {blog_topic}.",
    agent=blog_writer,
    async_execution=False,
    output_file="outputs/blog_{chroma_collection_name}.txt"  # Dynamically store output in the outputs folder
)

# Assemble the Crew with a sequential process
crew = Crew(
    agents=[data_retriever, blog_writer],
    tasks=[retrieval_task, writing_task],
    process=Process.sequential  # Ensures retrieval is completed before writing
)

# Function to generate the file path dynamically based on Chroma DB collection name
def generate_output_file_path(collection_name: str) -> str:
    # Construct the path with the dynamic file name using the collection name
    return os.path.join("outputs", f"blog_{collection_name}.txt")

# Execute the crew on a specified topic
if __name__ == "__main__":
    blog_topic = "Laila overview"  # Example blog topic
    collection_name = persistent_directory.split('/')[-1]  # Get the last part of the persistent directory for collection name
    output_file_path = generate_output_file_path(collection_name)  # Generate the file path based on collection name

    # Update the writing task to use the dynamically generated file path
    writing_task.output_file = output_file_path
    
    # Kick off the crew process
    result = crew.kickoff(inputs={'blog_topic': blog_topic})
    
    # Convert result to string
    result_text = str(result)  # Convert CrewOutput to string, or access specific attributes if necessary
    
    # Save the result to the dynamically generated output path
    with open(output_file_path, "w") as f:
        f.write(result_text)  # Write output to the dynamically named file
    
    print(f"Blog post written to {output_file_path}")
