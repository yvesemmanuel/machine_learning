"""Chatbot app front-end."""
import os
import streamlit as st
from agent import generate_agent_chain
from preprocessing import parse_pdf, text_to_docs, index_embedding


_API_KEY = os.getenv["OPENAI_API_KEY"]

st.title("The RAG Analyst 📊 ")
st.markdown(
    """ 
        #### Analyze your PDF files 📜 with `Conversational Buffer Memory`
    """
)

st.sidebar.markdown(
    """
    ### How to use:
    1. Upload some PDF file
    3. Perform Q&A

    **Note : PDF file content not stored in any form.**
    """
)

uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    filename = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    if pages:
        with st.expander("Show PDF extract corpus.", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]

        index = index_embedding(_API_KEY, pages)
        agent_chain = generate_agent_chain(_API_KEY, index)

        query = st.text_input(
            "**What's on your mind?**",
            placeholder="Ask me anything from {}".format(filename),
        )

        if query:
            with st.spinner("Generating Answer to your Query : `{}` ".format(query)):
                res = agent_chain.run(query)
                st.info(res, icon="🤖")

        with st.expander("History/Memory"):
            st.session_state.memory

with st.sidebar:
    linkedin_url = "https://www.linkedin.com/in/yvesemmanuel/"
    st.sidebar.markdown(
        f"If you encounter any issues using the RAG analyst tool, "
        "feel free to contact me on my [LinkedIn page]({linkedin_url}).",
        unsafe_allow_html=True,
    )
