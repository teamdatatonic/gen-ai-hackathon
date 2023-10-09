from google.cloud import bigquery, aiplatform
import os 

# PROJECT_ID = os.environ.get["PROJECT_ID"]
# DATASET_ID = os.environ.get["BQ_DATASET"]

PROJECT_ID = 'dt-gen-ai-hackathon-dev'
DATASET_ID = 'database_analytics_demo_v2'
# bigquery database

def biqquery_client():

    # Need to import env var s to use here and export the cleint into chains.py
    # Initialize the BigQuery client
    client = bigquery.Client(project=PROJECT_ID)

    # Initialise AI Platform
    aiplatform_client = aiplatform.init(project=PROJECT_ID)

    return client, aiplatform_client