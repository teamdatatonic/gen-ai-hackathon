from langchain.llms import VertexAI
from langchain import SQLDatabase
from tabulate import tabulate

from dt_gen_ai_analytics_helper.chains.chains import create_db_chain
from dt_gen_ai_analytics_helper.chains.chains import create_agent
from dt_gen_ai_analytics_helper.chains.chains import resummarise
from dt_gen_ai_analytics_helper.bigquery.bigquery import biqquery_client
from dt_gen_ai_analytics_helper.database.database import BigQueryDatabase

import dt_gen_ai_analytics_helper.view.view as demo_view
import gradio as gr
import json
import pandas

# Define environmental variables
PROJECT_ID = 'dt-gen-ai-hackathon-dev'
DATASET_ID = 'database_analytics_demo_v2'

biqquery_client()

llm = VertexAI(model_name='text-bison@001',
               temperature=0, max_output_tokens=1024)

db = BigQueryDatabase(project_id=PROJECT_ID, dataset_id=DATASET_ID)
session = db.create_session()

langchain_db = SQLDatabase(
    db.engine, schema=db.schema, sample_rows_in_table_info=0)



# Define a function to query the SQLDBChain
def query_database(question, llm=llm, db=langchain_db):
    # Call the SQLDBChain to get the answer based on the question
    output = create_db_chain(llm, db, question)

    chatbot_history = []

# get SQL query from natural language
    sql_query = output["intermediate_steps"][1]
    query_result = output["intermediate_steps"][3]
    nl_summary = output["result"]

    dataframe = pandas.DataFrame.from_records(query_result)
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
        markdown_table,
        sql_query
    )

# Define a function to summerize an sql result
def resummarise_sql(question):
    chatbot_history, markdown_table, sql_query= query_database(question=question)

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



# Create a Gradio interface 
iface = gr.Interface(
    fn= sql_agent,  # Function to execute when a query is received
    inputs="text",      # Input is a single text field
    outputs="text",     # Output will be a text response
    title="Analytics Worker Demo",
    description="Enter a question, and the system will query the database and provide an answer.",
)

# run app
if __name__ == "__main__":
    # Launch the Gradio interface on a specified port (e.g., 5000)
    iface.launch(share=True)