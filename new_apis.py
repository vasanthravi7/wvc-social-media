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
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64
from flask_cors import CORS
from os import scandir

app = Flask(__name__)

CORS(app)

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
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(max_tokens=300)

# Global variables for persistent_directory and chroma_db
persistent_directory = None
chroma_db = None
# Custom tool for querying Chroma DB
@tool
def query_chroma_db(query_text: str) -> str:
    """
    Retrieves relevant chunks from Chroma DB using a semantic search
    based on the input query_text.
    """
    results = chroma_db.similarity_search(query_text, k=10)
    formatted_results = "\n\n".join([doc.page_content for doc in results if hasattr(doc, 'page_content')])
    return formatted_results

# Define Agents
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

# Crew Definition
crew = Crew(
    agents=[data_retriever, blog_writer, instagram_converter, linkedin_converter],
    tasks=[retrieval_task, writing_task, instagram_task, linkedin_task],
    process=Process.sequential
)

# Helper Function: Generate File Path
def generate_output_file_path(collection_name: str, file_type: str) -> str:
    return os.path.join("outputs", f"{file_type}_{collection_name}.txt")


# Helper Function: Read File Content
def read_file_content(file_path: str) -> str:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None


# API: Generate Blog
@app.route('/generate_blog', methods=['POST'])
def generate_blog():
    try:
        data = request.get_json()
        blog_topic = data.get('blog_topic')
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')

        # Validate required parameters
        if not blog_topic or not folder_name or not file_name:
            return jsonify({"error": "Missing 'blog_topic', 'folder_name', or 'file_name' parameter"}), 400

        # Initialize persistent directory and Chroma DB
        global persistent_directory, chroma_db
        persistent_directory = os.path.join(folder_name, file_name)
        chroma_db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )

        # Prepare output file path
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)  # Ensure 'outputs' directory exists
        blog_output_file = os.path.join(output_dir, f"blog_{file_name}.txt")
        writing_task.output_file = blog_output_file

        # Define a specific crew for blog generation
        blog_crew = Crew(
            agents=[data_retriever, blog_writer],
            tasks=[retrieval_task, writing_task],
            process=Process.sequential
        )

        # Execute the blog generation process
        blog_crew.kickoff(inputs={'blog_topic': blog_topic, 'folder_name': folder_name, 'file_name': file_name})

        # Verify output file existence
        if not os.path.exists(blog_output_file):
            return jsonify({"error": "Blog output file not generated"}), 500

        # Read and return the content of the blog file
        blog_content = read_file_content(blog_output_file)
        return jsonify({
            "message": "Blog generated successfully",
            "blog_file": blog_output_file,
            "blog_content": blog_content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/generate_instagram', methods=['POST'])
def generate_instagram():
    try:
        data = request.get_json()
        blog_topic = data.get('blog_topic')
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')

        if not blog_topic or not folder_name or not file_name:
            return jsonify({"error": "Missing 'blog_topic', 'folder_name', or 'file_name' parameter"}), 400

    
        # Determine file paths dynamically
        blog_output_file = os.path.join("outputs", f"blog_{file_name}.txt")
        instagram_output_file = os.path.join("outputs", f"instagram_post_{file_name}.txt")

        # Check if the blog file exists
        if not os.path.exists(blog_output_file):
            return jsonify({"error": "Blog file not found. Generate the blog first."}), 400

        # Read the blog content from the file
        blog_content = read_file_content(blog_output_file)
        if not blog_content:
            return jsonify({"error": "Blog content could not be read from the file."}), 500

        # Use the Instagram agent to generate the Instagram content
        instagram_task.output_file = instagram_output_file

        # Define the Crew and run the task (use 'kickoff' for execution)
        instagram_crew = Crew(
            agents=[instagram_converter],
            tasks=[instagram_task],
            process=Process.sequential
        )

        # Execute the crew with the blog content as input
        instagram_crew.kickoff(inputs={"blog_content": blog_content, 'folder_name': folder_name, 'file_name': file_name})

        # Check if the Instagram output file is generated
        if not os.path.exists(instagram_output_file):
            return jsonify({"error": "Instagram output file not generated"}), 500

        # Read the Instagram content from the file
        instagram_content = read_file_content(instagram_output_file)

        # Return the Instagram content and file path
        return jsonify({
            "message": "Instagram post generated successfully",
            "instagram_file": instagram_output_file,
            "instagram_content": instagram_content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_linkedin', methods=['POST'])
def generate_linkedin():
    try:
        data = request.get_json()
        blog_topic = data.get('blog_topic')
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')

        if not blog_topic or not folder_name or not file_name:
            return jsonify({"error": "Missing 'blog_topic', 'folder_name', or 'file_name' parameter"}), 400

    

        # Determine file paths dynamically
        blog_output_file = os.path.join("outputs", f"blog_{file_name}.txt")
        linkedin_output_file = os.path.join("outputs", f"linkedin_post_{file_name}.txt")

        # Check if the blog file exists
        if not os.path.exists(blog_output_file):
            return jsonify({"error": "Blog file not found. Generate the blog first."}), 400

        # Read the blog content from the file
        blog_content = read_file_content(blog_output_file)
        if not blog_content:
            return jsonify({"error": "Blog content could not be read from the file."}), 500

        # Use the LinkedIn agent to generate the LinkedIn content
        linkedin_task.output_file = linkedin_output_file

        # Define the Crew and run the task (use 'kickoff' for execution)
        linkedin_crew = Crew(
            agents=[linkedin_converter],
            tasks=[linkedin_task],
            process=Process.sequential
        )

        # Execute the crew with the blog content as input
        linkedin_crew.kickoff(inputs={"blog_content": blog_content, 'folder_name': folder_name, 'file_name': file_name})

        # Check if the LinkedIn output file is generated
        if not os.path.exists(linkedin_output_file):
            return jsonify({"error": "LinkedIn output file not generated"}), 500

        # Read the LinkedIn content from the file
        linkedin_content = read_file_content(linkedin_output_file)

        # Return the LinkedIn content and file path
        return jsonify({
            "message": "LinkedIn post generated successfully",
            "linkedin_file": linkedin_output_file,
            "linkedin_content": linkedin_content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to recursively list all subfolders
def list_subfolders(base_folder="images"):
    subfolders = []
    for root, dirs, _ in os.walk(base_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            subfolders.append(subfolder_path)
    return subfolders

# Find the most relevant subfolder based on query
def find_relevant_subfolder(query, subfolders):
    folder_names = [os.path.basename(subfolder) for subfolder in subfolders]
    query_embedding = text_model.encode([query])
    subfolder_embeddings = text_model.encode(folder_names)
    similarities = cosine_similarity(query_embedding, subfolder_embeddings)
    best_match_idx = similarities.argmax()
    return subfolders[best_match_idx]  # Return full path of the best match

# Embed all images in the subfolder using CLIP
def embed_images_in_folder(folder_path):
    image_embeddings = []
    image_files = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_embeds = clip_model.get_image_features(**inputs)
            image_embeddings.append(image_embeds)
            image_files.append(image_path)
    image_embeddings = torch.vstack(image_embeddings)
    return image_files, image_embeddings

# Embed query and find the most relevant images
def get_query_embedding(query):
    inputs = clip_processor(text=query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs)
    return query_embedding

def find_top_n_images(query, folder, top_n=2):
    image_files, image_embeddings = embed_images_in_folder(folder)
    query_embedding = get_query_embedding(query)
    similarities = cosine_similarity(query_embedding.numpy(), image_embeddings.numpy())
    top_n_indices = similarities.argsort()[0][-top_n:][::-1]
    top_images = [image_files[i] for i in top_n_indices]
    return top_images

# Get the highest quality image
def get_highest_quality_image(images):
    best_image = None
    max_pixels = 0
    for image_path in images:
        image = Image.open(image_path)
        pixels = image.width * image.height
        if pixels > max_pixels:
            max_pixels = pixels
            best_image = image_path
    return best_image

# Flask API Endpoints
@app.route('/highest-quality-image', methods=['GET'])
def highest_quality_image():
    query = request.args.get('query')
    base_folder = "images"

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # Step 1: List all subfolders
    subfolders = list_subfolders(base_folder)
    
    # Step 2: Find the most relevant subfolder
    relevant_subfolder = find_relevant_subfolder(query, subfolders)
    
    # Step 3: Get top images and find the highest-quality one
    top_images = find_top_n_images(query, relevant_subfolder)
    highest_quality_image = get_highest_quality_image(top_images)

    if highest_quality_image:
        # Open the image and convert it to a base64 string
        with open(highest_quality_image, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            # Create a data URL for the image
            image_extension = highest_quality_image.split('.')[-1]
            image_url = f"data:image/{image_extension};base64,{base64_image}"
        return jsonify({"highest_quality_image": image_url})
    else:
        return jsonify({"error": "No images found"}), 404



@app.route('/get_files', methods=['GET'])
def get_files():
    """
    Optimized API endpoint to fetch file collections.
    """
    base_dir = request.args.get('data_type')

    if not os.path.exists(base_dir):
        return jsonify({"error": f"The folder '{base_dir}' does not exist."}), 404

    # Use scandir to iterate through directories faster
    files = [f"{entry.name}.pdf" for entry in scandir(base_dir) if entry.is_dir()]

    return jsonify(files)

if __name__ == '__main__':
    app.run(debug=True)
