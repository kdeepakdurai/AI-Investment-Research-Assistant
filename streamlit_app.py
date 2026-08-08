import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# Page configuration
st.set_page_config(
    page_title="AI Investment Research Assistant",
    page_icon="📈",
    layout="wide"
)


# Header
st.title("📈 AI Investment & Financial Research Assistant")
st.caption("Capstone Project 3")


# Check API key
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to the .env file.")
    st.stop()


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    google_api_key=api_key
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
prompt = st.chat_input(
    "Ask a financial research question..."
)


# Process user question
if prompt:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = llm.invoke(
                f"""
You are an AI Investment and Financial Research Assistant.

Answer the user's question clearly and professionally.

If the question is about finance or investing, explain
concepts in a simple and useful way.

Do not claim that you have current market data unless
current data has actually been provided.

User question:
{prompt}
"""
            )

            # Handle Gemini response format
            content = response.content

            if isinstance(content, str):
                answer = content

            elif isinstance(content, list):
                parts = []

                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)

                answer = "\n".join(parts)

            else:
                answer = str(content)

            # Display answer
            st.markdown(answer)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })