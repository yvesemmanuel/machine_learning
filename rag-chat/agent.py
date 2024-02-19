"""RAG agent module."""
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS


def generate_agent_chain(
    api_key: str,
    index: FAISS,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0
):
    qa_system = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api_key),
        chain_type="map_reduce",
        retriever=index.as_retriever(),
    )

    tools = [
        Tool(
            name="Q&A System",
            func=qa_system.run,
            description="Useful for answering questions about the PDF content.",
        )
    ]

    prefix = "Have a conversation with a human, answering questions based on the PDF content."
    suffix = "Begin your question with 'Question:' and then state your query."

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        llm=OpenAI(
            temperature=temperature, openai_api_key=api_key, model_name=model_name
        ),
        prompt=prompt,
    )

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
    )
