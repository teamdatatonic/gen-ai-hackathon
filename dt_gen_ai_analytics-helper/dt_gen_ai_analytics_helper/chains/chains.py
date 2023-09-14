from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.llms import VertexAI
from sqlalchemy.engine import create_engine
import os

from dt_gen_ai_analytics_helper.bigquery.bigquery import biqquery_client
from google.cloud import aiplatform

aiplatform.init()
# Import env variables
# PROJECT_ID = os.environ["PROJECT_ID"]
# DATASET_ID = os.environ["BQ_DATASET"]
PROJECT_ID = 'dt-gen-ai-hackathon-dev'
DATASET_ID = 'database_analytics_demo_v2'

def create_db_chain():
    """ Create a Q&A conversation chain using the VertexAI LLM.

    """
    biqquery_client()
    engine = create_engine(f"bigquery://{PROJECT_ID}/{DATASET_ID}")
   
   
    # We use the VertexAI LLM for the chain, however other models can be substituted here

    llm = VertexAI(model_name='text-bison@001',
               temperature=0, max_output_tokens=1024)


    # The chain is set to return the result of the SQL query and the SQL query used in generating an output
    # This allows for explainability behind model output.
    db = SQLDatabase(engine=engine)


    db_chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=True,
        use_query_checker=True
    )
    return db_chain



def qa(params) -> tuple[str, str, str]:
    """
    This function generates a response based on the given parameters.

    Args:
        params: The parameters for generating a response.

    Returns:
        A tuple containing the response, the sources used, and the full prompt.
    """

    query = params.user_message

    # Create the chain and keep the prompt for logging
    qa_chain = create_db_chain()


    # Generate the response
    response = qa_chain(inputs={"query": query})


    # Return the LLM answer, and list of sources used (formatted as a string)
    return response["result"]





