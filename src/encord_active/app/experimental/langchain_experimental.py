import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

from encord_active.app.common.utils import setup_page


def langchain_experimental():
    def render():
        setup_page()
        load_dotenv()

        llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        tools = load_tools(["serpapi", "llm-math"], llm=llm)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        query = st.text_input("Query")

        if query != "":
            result = agent.run(query)
            st.write(result)

    return render
