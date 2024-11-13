import streamlit as st

# Dummy data for dropdowns
pdf_files = ["PDF Document 1", "PDF Document 2", "PDF Document 3"]
social_media_options = ["Instagram Post", "LinkedIn Post"]

# Configure page
st.set_page_config(
    page_title="Professional Dashboard",
    page_icon="🌟",
    layout="centered",
)

# Sidebar for PDF selection
st.sidebar.header("📁 Document Selection")
selected_pdf = st.sidebar.selectbox("Choose a PDF document", options=pdf_files, help="Select a PDF document to base your query or post on.")
st.sidebar.markdown("---")  # Divider for sidebar sections

# Sidebar App Information
st.sidebar.header("ℹ️ App Information")
st.sidebar.write("This app allows you to interact with a PDF-based chatbot and manage social media posts with ease.")
st.sidebar.write("**How it works:** Select a PDF in the sidebar, and then use the tabs to either ask questions to the chatbot or create social media posts.")

# Main title
st.title("🌐 Professional Interaction Dashboard")

# Tabs for Chatbot and Social Media
tabs = st.tabs(["Chatbot", "Social Media"])

# Chatbot Tab
with tabs[0]:
    st.header("🤖 Chatbot Assistant")
    st.write("Get answers based on the content of the selected PDF document.")
    
    # User query input
    user_query = st.text_input("Enter your question", placeholder="Type your question here...")
    
    # Submit button
    if st.button("Get Response", key="chatbot"):
        if user_query and selected_pdf:
            # Placeholder for actual response processing
            dummy_response = f"Simulated response for '{user_query}' using {selected_pdf}."
            st.success(dummy_response)
        else:
            st.warning("Please enter a question and make sure a PDF is selected.")

# Social Media Tab
with tabs[1]:
    st.header("📱 Social Media Manager")
    st.write("Craft a post for either Instagram or LinkedIn based on the selected PDF content.")

    # Dropdown to select platform
    post_type = st.selectbox("Select Platform", options=social_media_options, help="Choose the platform to publish your post.")
    
    # User post query
    post_content = st.text_area(f"Enter content for {post_type}", placeholder="Write your content here...")

    # Submit button
    if st.button("Create Post", key="social_media"):
        if post_content and selected_pdf:
            # Placeholder for actual social media post logic
            dummy_social_response = f"Simulated post creation on {post_type} with content: '{post_content}' using {selected_pdf}."
            st.success(dummy_social_response)
        else:
            st.warning("Please enter post content and make sure a PDF is selected.")

# Style tweaks for a professional look
st.markdown("""
    <style>
        /* Center the main title */
        .css-18e3th9 {
            text-align: center;
        }
        
        /* Customize button style */
        .stButton>button {
            color: #fff;
            background-color: #4CAF50;
            padding: 8px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 1em;
        }

        /* Set a light background for tabs */
        .css-1q8dd3e {  /* Adjusts tab background */
            background-color: #f4f4f9;
            border-radius: 5px;
        }
        
        /* Style the sidebar section headers */
        .css-hxt7ib h2 {
            color: #333;
            font-size: 1.2em;
        }

        /* Adjust input box spacing */
        .css-1d391kg {
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
