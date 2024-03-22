import streamlit as st
from pandasai import SmartDataframe
#from pandasai.llm import OpenAI
from pandasai.llm import BambooLLM
from pandasai.responses.response_parser import ResponseParser
import pandas as pd
import pickle
import os
from pathlib import Path

def load_file(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        dataset = pickle.load(f)
        return dataset


@st.cache_data
def load_data(folder: str) -> pd.DataFrame:
    all_datasets = [load_file(file) for file in Path(folder).iterdir()]
    df = pd.concat(all_datasets)
    return df



class ResponseParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return
    
# --- User interface

st.write("# Chat with a data set with PandasAI")

df=load_data("./data")

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

query = st.text_area("🗣️ Query the Dataframe")
container = st.container()

if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": ResponseParser
        },
    )

    answer = query_engine.chat(query)