import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.financial_news_agent import search_financial_news
from app.agents.company_research_agent import research_company
from app.rag.pdf_rag import ask_report


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Investment Research Assistant",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📈 AI Investment Research Assistant")
st.caption("Capstone Project 3 | AI-Powered Financial Research")


# --------------------------------------------------
# API Key Check
# --------------------------------------------------

if not api_key:
    st.error("GEMINI_API_KEY is missing. Check your .env file.")
    st.stop()


# --------------------------------------------------
# Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    google_api_key=api_key
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🔎 Research Tools")

company = st.sidebar.text_input(
    "Company / Stock",
    value="NVIDIA"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Use the tools below to research a company, "
    "find financial news, or ask questions about "
    "the annual report."
)


# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Assistant",
    "🏢 Company Research",
    "📰 Financial News",
    "📄 Annual Report RAG"
])


# ==================================================
# TAB 1 - AI ASSISTANT
# ==================================================

with tab1:

    st.subheader("💬 AI Financial Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask a financial research question..."
    )

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing..."):

                response = llm.invoke(
                    f"""
You are an AI Investment and Financial Research Assistant.

Answer the user's question clearly and professionally.

Explain financial concepts simply when appropriate.

Do not invent current market data.

User question:
{prompt}
"""
                )

                content = response.content

                if isinstance(content, str):
                    answer = content

                elif isinstance(content, list):

                    parts = []

                    for item in content:

                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(
                                    item.get("text", "")
                                )

                        elif isinstance(item, str):
                            parts.append(item)

                    answer = "\n".join(parts)

                else:
                    answer = str(content)

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ==================================================
# TAB 2 - COMPANY RESEARCH
# ==================================================

with tab2:

    st.subheader("🏢 Company Research")

    st.write(
        "Research company information using the "
        "Company Research Agent."
    )

    if st.button(
        "🔍 Research Company",
        key="company_research"
    ):

        if not company.strip():

            st.warning("Enter a company name.")

        else:

            with st.spinner(
                f"Researching {company}..."
            ):

                try:

                    result = research_company(
                        company.strip()
                    )

                    st.success(
                        f"Research completed for {company}"
                    )

                    st.markdown("### Company Research")

                    st.write(result)

                except Exception as e:

                    st.error(
                        f"Company research failed: {e}"
                    )


# ==================================================
# TAB 3 - FINANCIAL NEWS
# ==================================================

with tab3:

    st.subheader("📰 Financial News")

    st.write(
        "Find recent financial news and information "
        "about the selected company."
    )

    if st.button(
        "📰 Search Financial News",
        key="financial_news"
    ):

        if not company.strip():

            st.warning("Enter a company name.")

        else:

            with st.spinner(
                f"Searching news about {company}..."
            ):

                try:

                    result = search_financial_news(
                        company.strip()
                    )

                    st.success(
                        f"News search completed for {company}"
                    )

                    st.markdown("### Financial News")

                    st.write(result)

                except Exception as e:

                    st.error(
                        f"Financial news search failed: {e}"
                    )


# ==================================================
# TAB 4 - PDF RAG
# ==================================================

with tab4:

    st.subheader("📄 Annual Report Research")

    st.write(
        "Ask questions about the NVIDIA annual report "
        "using Retrieval-Augmented Generation (RAG)."
    )

    st.info(
        "The current RAG database contains the NVIDIA "
        "annual report."
    )

    question = st.text_input(
        "Ask a question about the annual report",
        placeholder="Example: What are NVIDIA's main sources of revenue?"
    )

    if st.button(
        "📄 Search Annual Report",
        key="annual_report"
    ):

        if not question.strip():

            st.warning("Enter a question.")

        else:

            with st.spinner(
                "Searching the annual report..."
            ):

                try:

                    documents = ask_report(
                        question.strip()
                    )

                    if not documents:

                        st.warning(
                            "No relevant information found."
                        )

                    else:

                        st.markdown(
                            "### 📚 Relevant Report Information"
                        )

                        context = ""

                        for i, document in enumerate(
                            documents,
                            1
                        ):

                            st.markdown(
                                f"#### Source {i}"
                            )

                            st.write(
                                document.page_content
                            )

                            context += (
                                document.page_content
                                + "\n\n"
                            )

                        # Generate a concise answer
                        st.markdown(
                            "### 🤖 AI Analysis"
                        )

                        response = llm.invoke(
                            f"""
You are an investment research assistant.

Answer the question using ONLY the information
provided from the annual report below.

If the information is not available in the
provided context, clearly say that it is not
available.

Question:
{question}

Annual Report Context:
{context}
"""
                        )

                        content = response.content

                        if isinstance(content, str):
                            answer = content

                        elif isinstance(content, list):

                            parts = []

                            for item in content:

                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        parts.append(
                                            item.get(
                                                "text",
                                                ""
                                            )
                                        )

                                elif isinstance(item, str):
                                    parts.append(item)

                            answer = "\n".join(parts)

                        else:
                            answer = str(content)

                        st.write(answer)

                except Exception as e:

                    st.error(
                        f"Annual report search failed: {e}"
                    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AI Investment Research Assistant | "
    "Capstone Project 3"
)