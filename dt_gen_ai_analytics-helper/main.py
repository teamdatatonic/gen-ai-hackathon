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

    with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    with gr.Row():
                        # Load Datatonic logo as .svg
                        gr.Markdown(
                            """\
<svg width="177" height="24" viewBox="0 0 177 24" xmlns="http://www.w3.org/2000/svg"><path d="M14.5548 14.596H9.37748V9.404H0V14.596H9.37748V24H14.5548V14.596H23.9323V9.404H14.5548V14.596Z" fill="#2a5cff"></path><path d="M14.5548 0H9.37748V9.404H14.5548V0Z" fill="#2a5cff"></path><path d="M59.6168 6.3732C55.4116 6.3732 52.3637 8.42451 51.9375 11.8749H56.6305C56.8133 10.6942 57.7879 9.60525 59.6168 9.60525C61.5972 9.60525 62.5728 10.9415 62.5728 12.6518V12.931L58.2151 13.3658C54.9541 13.6769 51.3588 14.6403 51.3588 18.4955C51.3588 21.7594 53.918 23.9981 57.1183 23.9981C60.3186 23.9981 61.6891 22.4753 62.6334 20.454V23.6252H67.2345V12.9938C67.2345 9.20136 64.7663 6.37226 59.6168 6.37226V6.3732ZM62.5728 16.4451C62.5728 19.0877 61.3235 21.139 58.9462 21.139C57.3617 21.139 56.1124 20.1448 56.1124 18.4964C56.1124 16.5997 58.0323 16.2586 59.9218 16.0403L62.5728 15.7601V16.4451Z" fill="#2a5cff"></path><path d="M76.7895 10.2275H80.416V6.74523H76.7895V2.76725H72.1277V6.74616H69.0496V10.2284H72.1277V18.622C72.1277 21.9796 74.5656 23.627 77.8257 23.627H80.4463V20.1457H78.8315C77.4904 20.1457 76.7895 19.7109 76.7895 18.249V10.2275Z" fill="#2a5cff"></path><path d="M90.6838 6.3732C86.4786 6.3732 83.4308 8.42451 83.0046 11.8749H87.6975C87.8803 10.6942 88.8549 9.60525 90.6838 9.60525C92.6643 9.60525 93.6398 10.9415 93.6398 12.6518V12.931L89.2821 13.3658C86.0212 13.6769 82.4259 14.6403 82.4259 18.4955C82.4259 21.7594 84.985 23.9981 88.1853 23.9981C91.3856 23.9981 92.7561 22.4753 93.7004 20.454V23.6252H98.3016V12.9938C98.3016 9.20136 95.8333 6.37226 90.6838 6.37226V6.3732ZM93.6398 16.4451C93.6398 19.0877 92.3905 21.139 90.0133 21.139C88.4287 21.139 87.1795 20.1448 87.1795 18.4964C87.1795 16.5997 89.0993 16.2586 90.9888 16.0403L93.6398 15.7601V16.4451Z" fill="#2a5cff"></path><path d="M108.022 10.2275H111.648V6.74523H108.022V2.76725H103.36V6.74616H100.282V10.2284H103.36V18.622C103.36 21.9796 105.798 23.627 109.058 23.627H111.679V20.1457H110.064C108.723 20.1457 108.022 19.7109 108.022 18.249V10.2275Z" fill="#2a5cff"></path><path d="M121.926 6.3732C116.624 6.3732 113.303 10.0101 113.303 15.2016C113.303 20.3931 116.624 23.9991 121.926 23.9991C127.228 23.9991 130.55 20.3622 130.55 15.2016C130.55 10.041 127.228 6.3732 121.926 6.3732ZM121.926 20.7661C119.397 20.7661 118.056 18.6211 118.056 15.2016C118.056 11.7821 119.397 9.60618 121.926 9.60618C124.455 9.60618 125.796 11.7512 125.796 15.2016C125.796 18.652 124.455 20.7661 121.926 20.7661Z" fill="#2a5cff"></path><path d="M143.121 6.3732C140.226 6.3732 138.61 8.08246 137.849 10.1966V6.74616H133.217V23.6261H137.88V14.0199C137.88 11.0353 139.007 9.69896 140.927 9.69896C142.847 9.69896 143.974 11.0353 143.974 13.9581V23.6261H148.637V12.9319C148.637 9.20136 146.991 6.3732 143.121 6.3732H143.121Z" fill="#2a5cff"></path><path d="M157.046 6.74616H152.383V23.6261H157.046V6.74616Z" fill="#2a5cff"></path><path d="M157.137 0H152.323V4.47651H157.137V0Z" fill="#2a5cff"></path><path d="M172.107 17.6268C171.589 19.5853 170.492 20.767 168.572 20.767C166.165 20.767 164.763 18.7148 164.763 15.1716C164.763 11.6284 166.104 9.60712 168.572 9.60712C170.492 9.60712 171.559 11.0371 171.955 12.8092H176.647C175.947 8.86119 172.93 6.37414 168.572 6.37414C163.514 6.37414 160.009 9.88731 160.009 15.1716C160.009 20.4559 163.453 24 168.572 24C172.717 24 176.007 21.7613 176.8 17.6268H172.107Z" fill="#2a5cff"></path><path d="M37.2505 0H28.7188V9.26789H33.533V4.10355H37.3727C41.6082 4.10355 44.0764 6.83894 44.0764 11.813C44.0764 16.7872 41.6082 19.5226 37.3727 19.5226H33.533V14.3666H28.7188V23.6261H37.2505C44.2895 23.6261 49.0431 19.7718 49.0431 11.813C49.0431 3.85428 44.2895 0 37.2505 0Z" fill="#2a5cff"></path><path d="M38.5356 9.26789H33.5376V14.3666H38.5356V9.26789Z" fill="#2a5cff"></path></svg>"""
                        )
                        gr.Markdown(
                            "# Datatonic Analytics Assistant",
                            elem_classes="title right",
                        )

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
                    #     value="refresh", variant="primary"
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
    # clear_history.click()

if __name__ == "__main__":
    demo.launch(share=True)
