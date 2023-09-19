from dt_gen_ai_analytics_helper.bigquery.bigquery import biqquery_client
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent 
from sqlalchemy.engine import create_engine
from langchain.llms import VertexAI
from google.cloud import aiplatform
import gradio as gr
import os


# instatiate the bigquery and aiplatform client
client = biqquery_client
aiplatform.init()

# Define project env variables
PROJECT_ID = 'dt-gen-ai-hackathon-dev'
DATASET_ID = 'database_analytics_demo_v2'

# Define database engine and LLM 
engine = create_engine(f"bigquery://{PROJECT_ID}/{DATASET_ID}", pool_pre_ping=True)
llm = VertexAI(model_name='text-bison@001',
               temperature=0, max_output_tokens=1024)
db = SQLDatabase(engine=engine)


# Define our agent’s toolkit which will be used to answer the user question
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
qa_chain = create_sql_agent(
    llm=llm,
    db=db, 
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)


# Define a function to query the sql Agent 
def query_database(question):

    answer = qa_chain.run(question)

    return answer


# Create a Gradio interface
iface = gr.Interface(
    fn=query_database,  # Function to execute when a query is received
    inputs="text",      # Input is a single text field
    outputs="text",     # Output will be a text response
    title="Analytics Worker Demo",
    description="Enter a question, and the system will query the database and provide an answer.",
)

# Launch the Gradio interface on a specified port (e.g., 5000)
iface.launch(share=True)


# run app
if __name__ == "__main__":
    query_database()