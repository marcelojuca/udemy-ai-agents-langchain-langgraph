import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from backend.core import run_llm

load_dotenv()

# Custom CSS for LangChain-inspired theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --langchain-green: #10b981;
        --langchain-green-dark: #059669;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --background-white: #ffffff;
        --background-gray: #f9fafb;
        --border-gray: #e5e7eb;
    }
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main h1 {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .main h2 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.875rem;
        margin-bottom: 0.75rem;
    }
    
    .main h3 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background-white);
        border-right: 1px solid var(--border-gray);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .css-1d391kg .css-1v0mbdj h1 {
        color: var(--langchain-green);
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .css-1d391kg .css-1v0mbdj h2 {
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .css-1d391kg .css-1v0mbdj p {
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border: 2px solid var(--border-gray);
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.2s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--langchain-green);
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--langchain-green);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--langchain-green-dark);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: var(--background-white);
        border: 1px solid var(--border-gray);
        border-radius: 12px;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    .stChatMessage[data-testid="user"] {
        background-color: var(--background-gray);
        border-left: 4px solid var(--langchain-green);
    }
    
    .stChatMessage[data-testid="assistant"] {
        background-color: var(--background-white);
        border-left: 4px solid var(--text-secondary);
    }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--langchain-green);
    }
    
    /* Markdown styling */
    .main .markdown {
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .main .markdown strong {
        color: var(--langchain-green);
        font-weight: 600;
    }
    
    /* Profile picture styling */
    .profile-picture {
        border-radius: 50%;
        border: 3px solid var(--langchain-green);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-gray), transparent);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def create_profile_picture(name: str, size: int = 150) -> Image.Image:
    """Create a profile picture with user initials using Pillow."""
    # Create a new image with LangChain green background
    img = Image.new('RGB', (size, size), color='#10b981')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_size = size // 3
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get initials from name
    initials = ''.join([word[0].upper() for word in name.split()[:2]])
    
    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # Draw the text
    draw.text((x, y), initials, fill='white', font=font)
    
    return img


# Sidebar with user information
with st.sidebar:
    st.markdown("### 🚀 LangChain Assistant")
    
    # User information
    user_name = "John Doe"
    user_email = "john.doe@example.com"
    
    # Create and display profile picture using Pillow
    profile_img = create_profile_picture(user_name, size=120)
    
    # Convert PIL image to bytes for Streamlit
    img_buffer = BytesIO()
    profile_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Center the profile picture
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_buffer, width=120)
    
    # User information with better styling
    st.markdown(f"#### {user_name}")
    st.markdown(f"📧 {user_email}")
    
    # Custom divider
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Additional user info with icons
    st.markdown("**🟢 Status:** Active")
    st.markdown("**📅 Member Since:** January 2024")
    st.markdown("**🕒 Last Login:** Today")
    
    # Add some spacing
    st.markdown("")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sessions", "24", "↗️ 3")
    with col2:
        st.metric("Queries", "156", "↗️ 12")

# Main content area with LangChain styling
st.markdown("# 🤖 LangChain Document Assistant")
st.markdown("### Powered by Advanced AI and Document Processing")

# Add some description
st.markdown("""
Welcome to your intelligent document assistant! Ask questions about your documents and get 
comprehensive answers with source citations. This assistant leverages LangChain's powerful 
framework to provide accurate and contextual responses.
""")

# Enhanced prompt input
st.markdown("#### 💬 Ask a Question")
prompt = st.text_input(
    "Enter your question about the documents", 
    placeholder="e.g., What are the main topics discussed in the documents?",
    help="Type your question and press Enter to get an AI-powered response with source citations."
)

if (
    ("user_prompt_history" not in st.session_state)
    and ("chat_answer_history" not in st.session_state)
    and ("chat_history" not in st.session_state)
):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(sources_urls: set[str]) -> str:
    if not sources_urls:
        return ""
    sources_list = list(sources_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("🤔 Analyzing your question..."):

        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# Display chat history with enhanced styling
if st.session_state["chat_answer_history"]:
    st.markdown("---")
    st.markdown("### 💬 Conversation History")
    
    for i, (generated_response, user_query) in enumerate(zip(
        st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]
    )):
        # User message
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_query}")
        
        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(f"**Assistant:** {generated_response}")
        
        # Add separator between conversations
        if i < len(st.session_state["chat_answer_history"]) - 1:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
