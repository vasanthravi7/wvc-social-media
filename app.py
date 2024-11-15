import streamlit as st
import requests
import time

# Dummy data for PDF dropdowns
pdf_files = ["World-Vision-Canada-FY23-Annual-Results-Report.pdf"]

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

# Main title
st.title("🌐 PDF Dashboard")

# Initialize session state variables to manage the UI state
if "show_query_input" not in st.session_state:
    st.session_state["show_query_input"] = True  # Controls the visibility of the content query input box
if "user_content_query" not in st.session_state:
    st.session_state["user_content_query"] = ""
if "generated_content" not in st.session_state:
    st.session_state["generated_content"] = {}

# Tabs for Chatbot and Social Media
tabs = st.tabs(["Chatbot", "Social Media"])

# Chatbot Tab
with tabs[0]:
    st.header("🤖 Chatbot Assistant")
    st.write("Get answers based on the content of the selected PDF document.")
    
    # Reduced size for user query input
    user_query = st.text_input("Enter your question", placeholder="Type your question here...")

    # Submit button for chatbot
    if st.button("Get Response", key="chatbot"):
        if user_query and selected_pdf:
            # Loading indicator while waiting for response
            with st.spinner("Fetching response... Please wait"):
                try:
                    # Simulate a delay for the API request
                    time.sleep(2)

                    # API call to chatbot service (replace with actual API call)
                    response = requests.post("http://127.0.0.1:5000/chat", json={"query": user_query})
                    chatbot_response = response.json().get("response", "No response from API")
                    
                    # Display chatbot response with larger, styled text area
                    st.markdown(f"<div style='background-color:#e0f7fa;padding:20px;border-radius:10px;'>"
                                f"<strong>Response:</strong><br><p style='font-size:20px;color:#000;'>{chatbot_response}</p>"
                                f"</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error connecting to the chatbot API: {e}")
        else:
            st.warning("Please enter a question and make sure a PDF is selected.")

# Social Media Tab with three sections, shown only if query is provided
with tabs[1]:
    st.header("📱 Social Media Manager")
    st.write("Craft posts based on the selected PDF content.")

    # Show content query input box only if `show_query_input` is True
    if st.session_state["show_query_input"]:
        # Reduced size for content query input
        user_content_query = st.text_input("Enter content query for posts", placeholder="Type your content query here...")
        
        # Generate content button to trigger content generation
        if st.button("Generate Content"):
            if user_content_query and selected_pdf:
                with st.spinner("Generating content... Please wait"):
                    try:
                        # Simulate a delay for the API request
                        time.sleep(2)

                        # API call to social media content generator (replace with actual API call)
                        response = requests.post("http://127.0.0.1:5000/generate_output", json={"blog_topic": user_content_query})
                        content = response.json()
                        st.session_state["generated_content"] = {
                            "blog": content.get("blog_content", "No blog content generated"),
                            "instagram": content.get("instagram_content", "No Instagram content generated"),
                            "linkedin": content.get("linkedin_content", "No LinkedIn content generated"),
                        }
                        st.session_state["show_query_input"] = False  # Hide the input box after generating content
                    except Exception as e:
                        st.error(f"Error connecting to the social media API: {e}")
            else:
                st.warning("Please enter a content query and make sure a PDF is selected.")
    
    # Show the generated content sections if content has been generated
    if not st.session_state["show_query_input"]:
        # Display generated content for each section
        sm_sections = st.tabs(["Blog Writer", "Instagram Post", "LinkedIn Post"])

        # Blog Writer Section
        with sm_sections[0]:
            st.subheader("📝 Blog Writer")
            st.text_area("Generated Blog Content", value=st.session_state["generated_content"]["blog"], height=700, key="blog_content")

        # Instagram Post Section
        with sm_sections[1]:
            st.subheader("📸 Instagram Post")
            st.text_area("Generated Instagram Content", value=st.session_state["generated_content"]["instagram"], height=700, key="instagram_content")
            
            if st.button("Instagram Post", key="insta_post"):
                st.success(f"Instagram post created with content: '{st.session_state['generated_content']['instagram']}' using {selected_pdf}.")

        # LinkedIn Post Section
        with sm_sections[2]:
            st.subheader("🔗 LinkedIn Post")
            st.text_area("Generated LinkedIn Content", value=st.session_state["generated_content"]["linkedin"], height=700, key="linkedin_content")
            
            if st.button("LinkedIn Post", key="linkedin_post"):
                st.success(f"LinkedIn post created with content: '{st.session_state['generated_content']['linkedin']}' using {selected_pdf}.")

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
            padding: 12px 30px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 1.1em;
        }

        /* Set a light background for tabs */
        .css-1q8dd3e {  /* Adjusts tab background */
            background-color: #f4f4f9;
            border-radius: 5px;
        }
        
        /* Style the sidebar section headers */
        .css-hxt7ib h2 {
            color: #333;
            font-size: 1.3em;
        }

        /* Adjust input box spacing */
        .css-1d391kg {
            margin-bottom: 20px;
        }

        /* Chatbot response styling */
        .chatbot-response {
            background-color: #e0f7fa;
            padding: 20px;
            border-radius: 10px;
            font-size: 20px;
            color: #000;  /* Set response text color to black */
            line-height: 1.6;
        }

        /* Increase the size of input fields */
        textarea {
            font-size: 16px;
        }

        /* Increase tab content font size */
        .css-1p8virf {
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)
