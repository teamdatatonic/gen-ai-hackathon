from langchain.llms import VertexAI
from langchain import SQLDatabase
from tabulate import tabulate

from dt_gen_ai_analytics_helper.chains.chains import create_db_chain
from dt_gen_ai_analytics_helper.chains.chains import create_agent
from dt_gen_ai_analytics_helper.chains.chains import resummarise
from dt_gen_ai_analytics_helper.chains.chains import pandas_agent
from dt_gen_ai_analytics_helper.bigquery.bigquery import biqquery_client
from dt_gen_ai_analytics_helper.database.database import BigQueryDatabase
import matplotlib.pyplot as plt

import dt_gen_ai_analytics_helper.view.view as demo_view
from sqlalchemy import create_engine, text
import gradio as gr
import json
import pandas as pd

import random
import time

# Define environmental variables
PROJECT_ID = 'dt-gen-ai-hackathon-dev'
DATASET_ID = 'database_analytics_demo_v2'

# Define LLM andBgquery Database connection 
biqquery_client()
llm = VertexAI(model_name='text-bison@001',
               temperature=0, max_output_tokens=1024)

db = BigQueryDatabase(project_id=PROJECT_ID, dataset_id=DATASET_ID)
session = db.create_session()

conn = db.create_connection()

langchain_db = SQLDatabase(
    db.engine, schema=db.schema, sample_rows_in_table_info=0)



# Define a function to query the SQLDBChain
def query_database(question, llm=llm, db=langchain_db):
    # Call the SQLDBChain to get the answer based on the question
    output = create_db_chain(llm, db, question)
 
    chatbot_history = []

    #  get SQL query from natural language
    sql_query = output["intermediate_steps"][1]
    query_result = output["intermediate_steps"][3]
    nl_summary = output["result"]


    
    dataframe = pd.DataFrame.from_records(query_result)
    # dataframe_json = dataframe.to_json()

    # dataframe_json = dataframe.to_json()
    markdown_table = tabulate(dataframe, headers='keys', tablefmt='pipe')

    # markdown_table = dataframe.to_markdown(index=False, floatfmt=".2f")

    chatbot_history.append(
        (
            f"{question}",
            f"{nl_summary}",
            f"{markdown_table}"
        )
    )


    return (
        chatbot_history,
        dataframe,
        sql_query
    )

# Define a function to summerize an sql result
def resummarise_sql(question):
    chatbot_history, markdown_table, sql_query= query_database(question=question)

    # dataframe = pd.read_json(dataframe_json)

    response = resummarise(query=question,
                            initial_query=sql_query,
                            initial_summary=chatbot_history[0][1],
                            initial_data=markdown_table,
                            llm=llm)
    
    chatbot_history.append(
        (
            f"{question}",
            f"{response}",
        )
    )

    return response


# Define an sql agent with SQLToolKit
def sql_agent(question):
    response = create_agent(llm=llm, db=langchain_db, question=question)

    return response

# Define an pandas agent and pass in a dataframe
def pd_agent(question):

    chatbot_history, dataframe, sql_query = query_database(question=question)
    df = pd.read_sql(sql_query, conn, index_col=None)

    print(df)
 
    response = pandas_agent(question=question, llm=llm, df=df)

    # check if response is a plot, then render that in UI else return response
    return response


# Gradio chatbot and interface
with gr.Blocks(title="Analytics Assistant") as demo:
    chatbot = gr.Chatbot()

    with gr.Tab("Ask a question:"):
        # Create a textbox for user questions
        msg = gr.Textbox(show_label=False)

        with gr.Row().style(equal_height=False):
            with gr.Column(scale=3):
                with gr.Row():
                    send_message = gr.Button(
                        value="Submit", variant="primary"
                    ).style(size="sm")
                    clear = gr.ClearButton([msg, chatbot])
                    # clear_history = gr.Button(
                    #     value="Clear History", variant="secondary"
                    # ).style(size="sm")

    def respond(question, chat_history):
        bot_message = pd_agent(question)
        chat_history.append((question, bot_message))
        time.sleep(2)
        return "", chat_history
    

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_message.click(
                respond, [msg, chatbot], [msg, chatbot]
            )

if __name__ == "__main__":
    demo.launch(share=True)
