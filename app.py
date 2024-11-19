import streamlit as st
import requests
import base64
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

# Function to simulate an image API that returns base64 image data
def get_base64_image_from_api(query):
    try:
        response = requests.get(f"http://127.0.0.1:5000/highest-quality-image?query={query}")
        if response.status_code == 200:
            image_data = response.json().get("highest_quality_image", "")
            return image_data
        else:
            return ""
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return ""

# Function to display image first and then content
def display_image_and_content(base64_image_url, content):
    if base64_image_url:
        st.image(base64_image_url, use_container_width=True)
    else:
        st.info("No image available for the query.")
    st.write("### Content")
    st.write(content)

# Tabs for Chatbot and Social Media
tabs = st.tabs(["Chatbot", "Social Media"])

# Chatbot Tab
with tabs[0]:
    st.header("🤖 Chatbot Assistant")
    st.write("Get answers based on the content of the selected PDF document.")
    
    # User query input for Chatbot
    user_query = st.text_input("Enter your question", placeholder="Type your question here...")

    # Submit button for chatbot
    if st.button("Get Response", key="chatbot"):
        if user_query and selected_pdf:
            with st.spinner("Fetching response... Please wait"):
                try:
                    # Simulate a delay for the API request
                    time.sleep(2)

                    # API call to chatbot service (replace with actual API call)
                    response = requests.post("http://127.0.0.1:5000/chat", json={"query": user_query})
                    chatbot_response = response.json().get("response", "No response from API")
                    
                    # Display only content for Chatbot
                    st.write("### Good to See Your Query Response")
                    st.write(chatbot_response)

                except Exception as e:
                    st.error(f"Error connecting to the chatbot API: {e}")
        else:
            st.warning("Please enter a question and make sure a PDF is selected.")

# Social Media Tab with three sections, shown only if query is provided
with tabs[1]:
    st.header("📱 Social Media Manager")
    st.write("Craft posts based on the selected PDF content.")

    # Show content query input box only if show_query_input is True
    if st.session_state["show_query_input"]:
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

                        # Store generated content in session state
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
    
    # Show the generated content and image
    if not st.session_state["show_query_input"]:
        # Get base64 image from the image API
        base64_image_url = get_base64_image_from_api(user_content_query)

        # Display generated content and image selectively
        sm_sections = st.tabs(["Blog Writer", "Instagram Post", "LinkedIn Post"])

        # Blog Writer Section
        with sm_sections[0]:
            st.subheader("📝 Blog Writer")
            blog_content = st.session_state["generated_content"]["blog"]
            st.write("### Content")
            st.write(blog_content)

        # Instagram Post Section
        with sm_sections[1]:
            st.subheader("📸 Instagram Post")
            instagram_content = st.session_state["generated_content"]["instagram"]
            display_image_and_content(base64_image_url, instagram_content)

        # LinkedIn Post Section
        with sm_sections[2]:
            st.subheader("🔗 LinkedIn Post")
            linkedin_content = st.session_state["generated_content"]["linkedin"]
            display_image_and_content(base64_image_url, linkedin_content)
