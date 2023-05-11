import streamlit as st
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

from encord_active.app.common.state import get_state
from encord_active.app.common.utils import setup_page


def magic_commands():
    def render():
        setup_page()

        llm = OpenAI(api_token="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")
        pandas_ai = PandasAI(llm)

        magic_prompt = st.text_input("What do you want to get?")
        df = get_state().merged_metrics.copy().drop(["tags"], axis=1)

        pandas_ai.run(df, prompt=magic_prompt)

        st.write("hello")

    return render
