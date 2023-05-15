import configparser
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import openai
import pandas as pd
import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.app.common.utils import setup_page
from encord_active.lib.metrics.metadata import fetch_metrics_meta

image_id = "image_id"
data_type = "data_type"
object_id = "object_id"
START_CODE_TAG = "START_CODE"
END_CODE_TAG = "END_CODE"
TOTAL_TRIAL = 5

instruct_message = """Today is {today_date}.
You are provided with a pandas dataframe (df) with {df_num_rows} rows and {df_num_cols} columns.
This is the result of `print(df.head())`:
{df_head}.
{image_id} represents the name of each image.
{object_id} represents the name of each object, if it is None, it means that row represents an image.
{data_type} represent weather the values in other columns belong to the image or object in that image.

Return the python code (do not import anything) and make sure to prefix the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG} exactly to get the answer to the following question. The code script between prefix {START_CODE_TAG} and suffix {END_CODE_TAG} should be directly evaluated with python's builtin exec method:
"""

error_correction_message = """
Today is {today_date}.
You are provided with a pandas dataframe (df) with {df_num_rows} rows and {df_num_cols} columns.
This is the result of `print(df.head())`:
{df_head}.
{image_id} represents the name of each image.
{object_id} represents the name of each object, if it is None, it means that row represents an image.
{data_type} represent weather the values in other columns belong to the image or object in that image.

The user asked the following question:
{question}

You generated this python code:
{code}

It fails with the following error:
{error_returned}

Correct the python code and return a new python code (do not import anything) that fixes the above mentioned error. Do not generate the same code again.
Make sure to prefix the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG} exactly.
"""


def magic_commands():
    def _get_chatgpt_response(instruct: str, question: str, start_code_tag: str, end_code_tag: str) -> str:
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent / "keys.ini")
        openai.api_key = config.get("KEY", "OPENAI_API_KEY")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant which provides a python code that can be used to get the answer to the questions.",
            },
            {"role": "user", "content": instruct},
        ]
        if question != "":
            messages.append({"role": "user", "content": question})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        result = response["choices"][0]["message"]["content"]
        result = result.replace(start_code_tag, "")
        result = result.replace(end_code_tag, "")
        return result

    def _capture_return_object(code_to_run: str, df: pd.DataFrame) -> Any:
        exec(code_to_run)
        lines = code_to_run.strip().split("\n")
        last_line = lines[-1].strip()
        if last_line.startswith("print(") and last_line.endswith(")"):
            last_line = last_line[6:-1]

        if last_line == "plt.show()":
            return plt.subplots()[0]  # this does not work for now
        try:
            return eval(last_line)
        except Exception:  # pylint: disable=W0718
            return None

    def render():
        setup_page()

        all_metrics = sorted([metric for metric in fetch_metrics_meta(get_state().project_paths)])

        df = get_state().merged_metrics.copy()

        st.write("Top 5 rows")
        st.dataframe(df.head(5))
        magic_prompt = st.text_input("What do you want to get?")

        # Make it a bit more verbose for the LLM
        df[data_type] = df.index.map(lambda x: "image" if len(x.split("_")) == 3 else "object")
        df[image_id] = df.index.map(lambda x: "_".join(x.split("_")[:3]))
        df[object_id] = df.index.map(lambda x: None if len(x.split("_")) == 3 else x.split("_")[-1])

        if magic_prompt != "":
            code_to_run = _get_chatgpt_response(
                instruct=instruct_message.format(
                    today_date=date.today(),
                    df_num_rows=df.shape[0],
                    df_num_cols=df.shape[1],
                    df_head=df.head(),
                    all_metrics=all_metrics,
                    image_id=image_id,
                    object_id=object_id,
                    data_type=data_type,
                    START_CODE_TAG=START_CODE_TAG,
                    END_CODE_TAG=END_CODE_TAG,
                ),
                question=magic_prompt,
                start_code_tag=START_CODE_TAG,
                end_code_tag=END_CODE_TAG,
            )

            count = 0
            valid_code = False
            while count < TOTAL_TRIAL:
                try:
                    exec(code_to_run, {"pd": pd, "df": df})
                    valid_code = True
                    break
                except Exception as e:
                    count += 1
                    code_to_run = _get_chatgpt_response(
                        instruct=error_correction_message.format(
                            today_date=date.today(),
                            df_num_rows=df.shape[0],
                            df_num_cols=df.shape[1],
                            df_head=df.head(),
                            all_metrics=all_metrics,
                            image_id=image_id,
                            object_id=object_id,
                            data_type=data_type,
                            question=magic_prompt,
                            code=code_to_run,
                            error_returned=e,
                            START_CODE_TAG=START_CODE_TAG,
                            END_CODE_TAG=END_CODE_TAG,
                        ),
                        question="",
                        start_code_tag=START_CODE_TAG,
                        end_code_tag=END_CODE_TAG,
                    )

            if valid_code:
                st.write(f"Generated code for this prompt (tried {count+1} times):")
                st.code(code_to_run, language="python", line_numbers=True)

                return_object = _capture_return_object(code_to_run, df)

                if return_object is None:
                    st.write("The generated code could not produce output")
                else:
                    st.write(return_object)
            else:
                st.write("A code could not be generated for this prompt")
        else:
            st.write("please provide a prompt to see the result")

    return render
