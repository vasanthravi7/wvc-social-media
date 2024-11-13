import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
import os
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_chroma import Chroma
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


# Step 1: Define function to extract TOC and split text
def extract_toc_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    toc_content = ""
    
    # Extract text from the first 5 pages (assuming TOC is in these pages)
    for page_num in range(5):  # First 5 pages
        page = doc[page_num]
        toc_content += page.get_text()

    # Create the prompt template to check for a table of contents
    prompt_template = """
    The following text may contain a Table of Contents (TOC). Please analyze the content and determine if a TOC is present. whole toc content as below sample format. Format the start and end page number as integer. 
    Capital letters are the main headings, and lower case letters are subheadings. Titles with subheadings shouldn't be set to the same page number.
    
    TOC is found, return it in the following format without any additional text and without any \n values ::
    [
        ("SECTION NAME", start_page_number, end_page_number),
        ...
    ]
    If no TOC is present, return "No".

    Text to analyze:
    {toc_content}

    Answer:
    """

    # Create the PromptTemplate with the extracted TOC content
    prompt = PromptTemplate(input_variables=["toc_content"], template=prompt_template)

    # Create an LLMChain with the prompt and model
    chain = prompt | llm

    # Run the chain with the extracted TOC content
    response = chain.invoke({"toc_content": toc_content})
    print(response)
    if api_type == 1:
        toc_content = response.content
    else:
        toc_content = response
    toc_list = ast.literal_eval(toc_content)
    doc.close()
    
    return toc_list


# Step 2: Function to process PDF and add text to Chroma DB
def process_pdf(pdf_path, toc_list):
    # Create an empty DataFrame to store the extracted content
    df = pd.DataFrame(columns=["Topic", "Content", "Start Page", "End Page"])
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Extract content based on TOC
    for topic, start_page, end_page in toc_list:
        content = ""
        for page_num in range(start_page - 1, end_page):  # Page numbering in PyMuPDF starts at 0
            page = doc[page_num]
            content += page.get_text()  # Extract text from the page and accumulate for the topic
        
        # Create a new DataFrame row with the topic, content, start and end page
        new_row = pd.DataFrame({
            "Topic": [topic],
            "Content": [content],
            "Start Page": [start_page],
            "End Page": [end_page]
        })
        
        # Append the new row using pd.concat
        df = pd.concat([df, new_row], ignore_index=True)

    doc.close()
    
    # Text Splitter to break content into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Chroma vector store initialization (this will store data in the directory you specify)
    persist_directory = f"./chroma_db/{os.path.splitext(os.path.basename(pdf_path))[0]}"  # Create a folder per file
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Process the DataFrame rows and store in Chroma DB
    def process_row(row):
        topic = row["Topic"]
        content = row["Content"]
        
        # Split the content into chunks
        chunks = text_splitter.split_text(content)
        
        # Add the topic to every chunk
        chunks_with_topic = [f"{topic}: {chunk}" for chunk in chunks]
        
        return chunks_with_topic

    # Process each row and store results in Chroma DB
    for idx, row in df.iterrows():
        chunks_with_topic = process_row(row)
        
        # Store chunks with metadata (topic, pages) in Chroma
        for chunk in chunks_with_topic:
            db.add_texts([chunk], metadatas=[{"topic": row["Topic"], "start_page": row["Start Page"], "end_page": row["End Page"]}])

    # Save the Chroma vector store (it persists automatically in the specified directory)
    print(f"Chroma DB for {os.path.basename(pdf_path)} stored at: {persist_directory}")

# Step 3: Loop through all PDF files in the 'data' folder
data_folder = 'data'
pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]

# Process each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(data_folder, pdf_file)
    print(f"Processing {pdf_path}...")
    
    # Step 3.1: Extract TOC
    toc_list = extract_toc_from_pdf(pdf_path)
    print(toc_list)
    if toc_list != "No":
        # Step 3.2: Process PDF and add to Chroma DB
        process_pdf(pdf_path, toc_list)
    else:
        print(f"No TOC found in {pdf_path}, skipping...")

