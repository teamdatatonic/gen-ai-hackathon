from langchain.llms import VertexAI
from langchain import SQLDatabase

from dt_gen_ai_analytics_helper.chains.chains import create_db_chain
from dt_gen_ai_analytics_helper.bigquery.bigquery import biqquery_client
from dt_gen_ai_analytics_helper.database.database import BigQueryDatabase

import dt_gen_ai_analytics_helper.view.view as demo_view
import gradio as gr

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
    answer = create_db_chain(llm, db, question)


    return answer

# Create a Gradio interface
iface = gr.Interface(
    fn=query_database,  # Function to execute when a query is received
    inputs="text",      # Input is a single text field
    outputs="text",     # Output will be a text response
    title="Analytics Worker Demo",
    description="Enter a question, and the system will query the database and provide an answer.",
)

# run app
if __name__ == "__main__":
    # Launch the Gradio interface on a specified port (e.g., 5000)
    iface.launch(share=True)