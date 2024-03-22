# Interact with Panda dataframe with natural language

[PandasAI](https://docs.pandas-ai.com/en/latest/) is a Python library that makes it easy to ask questions to our data in natural language.
It uses generative AI model to understand and interpret natural language queries and translate them into python code and SQL queries.

## Creation of the app

1. Get the Dataset from https://github.com/Fraud-Detection-Handbook/simulated-data-transformed.git under a data folder.
1. Start python virtual env:

    ```sh
    python -m venv .venv
    source .venv/Scripts/activate
    ```
1. Install needed libraries: `pip install -r requirements.txt`
1. Create a `.env` file with the API KEY needed to access OpenAI or Anthropic
1. Create a streamlit app

### Streamlit app structure

* The query to the Pandas dataset is via pandasAI's [SmartDataframe](https://docs.pandas-ai.com/en/latest/getting-started/#smartdataframe) which use a config to access to the llm

    ```python
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": ResponseParser
        },
    )

    answer = query_engine.chat(query)
    ```

## Execute the demo

* Start the app

    ```
    streamlit run App.py
    ```

* In the query entry text enter: "count the number of rows"
* "Get the top 10 CUSTOMER_ID with the largest fraud amount (a fraud being TX_FRAUD=1)"
* "Plot the amount of fraud for the top 10 CUSTOMER_ID"
* "Plot the distribution of transaction amount for fraud versus non-fraud transactions"

## Other implementation

* Using Pandas Dataframe LangChain agent
* LlamaIndex query engine


