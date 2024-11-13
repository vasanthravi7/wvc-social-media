import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
import os
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, OpenAI
# from openai import AzureOpenAI 
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()
# openai_api = OpenAI()
azure_endpoint = os.getenv("OPENAI_API_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
api_version = "2023-03-15-preview"
deployment_name = "gpt-4o"  # Make sure this matches your deployment name in Azure
# Initialize OpenAI embeddings
embeddings = AzureOpenAIEmbeddings( 
    azure_endpoint=azure_endpoint,
    # api_key=api_key,
    # api_version=api_version,
    )

# Initialize LangChain's AzureOpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version,
    model=deployment_name,
    temperature=1,
    max_tokens=300,
)

# Open the PDF file
pdf_path = 'data/World-Vision-Canada-FY23-Annual-Results-Report.pdf'  # Replace with the path to your PDF file
doc = fitz.open(pdf_path)

# Step 1: Extract text from the first 5 pages (assuming TOC is in these pages)
toc_content = ""
for page_num in range(5):  # First 5 pages
    page = doc[page_num]
    toc_content += page.get_text()
# print(toc_content)

# Create the prompt template to check for a table of contents
prompt_template = """
The following text may contain a Table of Contents (TOC). Please analyze the content and determine if a TOC is present. whole toc content as below sample format. Fromat the start and end page number as integer don't do any hallucinated values this page number is very important. if title has multiple subtiles don't make that page number seperate.
understand in a table of content Captial letters are the main headings under that title if any lower case letters are sub headings. so titles with subheadings don't set as same page number. Example PROGRESS AND CHANGE start at 24 but under that are subheadings so it consider as start and end as before the subheadings start.
 TOC is found, return it in the following format without any additional text and, also dont't add any \n values ::
[
    ("SECTION NAME", start_page_number, end_page_number),
    ...
]
If no TOC is present, return "No".

Text to analyze:

{toc_content}

Answer:
"""

# Create a PromptTemplate with the prompt and `toc_content` variable
prompt = PromptTemplate(input_variables=["toc_content"], template=prompt_template)

# Create an LLMChain with the prompt and model
chain = prompt | llm
# # Run the chain with the extracted TOC content
response = chain.invoke({"toc_content": toc_content})
toc_content = response.content
toc_list = ast.literal_eval(toc_content)
doc.close()

# Create an empty DataFrame to store the extracted content
df = pd.DataFrame(columns=["Topic", "Content", "Start Page", "End Page"])

# Open the PDF file
pdf_path = 'data/World-Vision-Canada-FY23-Annual-Results-Report.pdf'  # Replace with the path to your PDF file
doc = fitz.open(pdf_path)
for topic, start_page, end_page in toc_list:
    print(topic, start_page, end_page)
    content = ""
    for page_num in range(start_page - 1, end_page):  # Page numbering in PyMuPDF starts at 0
        page = doc[page_num]
        content += page.get_text()  # Extract text from the page and accumulate for the topic
    
    # Create a new DataFrame row with the topic, content, start and end page
    new_row = pd.DataFrame({"Topic": [topic], 
                            "Content": [content], 
                            "Start Page": [start_page], 
                            "End Page": [end_page]})
    
    # Append the new row using pd.concat
    df = pd.concat([df, new_row], ignore_index=True)

# Close the PDF file
doc.close()

# Display the DataFrame
print(df)

# Text Splitter to break content into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Chroma vector store initialization (this will store data in the directory you specify)
persist_directory = "./chroma_db"  # The directory where the database will be saved
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Process the DataFrame rows
def process_row(row):
    topic = row["Topic"]
    content = row["Content"]
    
    # Split the content into chunks first
    chunks = text_splitter.split_text(content)

    # Add the topic to every chunk
    chunks_with_topic = [f"{topic}: {chunk}" for chunk in chunks]

    # Return the chunks with the topic added
    return chunks_with_topic

# Process each row and store results in Chroma DB
for idx, row in df.iterrows():
    chunks_with_topic = process_row(row)
    
    # Store chunks with metadata (topic, pages) in Chroma
    for chunk in chunks_with_topic:
        db.add_texts([chunk], metadatas=[{"topic": row["Topic"], "start_page": row["Start Page"], "end_page": row["End Page"]}])

# Save the Chroma vector store (No need for persist())
# Chroma automatically persists in the specified directory. You can access it later.
print(f"Chroma DB stored at: {persist_directory}")

