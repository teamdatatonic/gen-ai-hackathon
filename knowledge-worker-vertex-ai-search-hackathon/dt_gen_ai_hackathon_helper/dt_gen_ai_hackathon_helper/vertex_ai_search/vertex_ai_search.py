import http.client
import json
import logging
import subprocess
from typing import Optional

import requests
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine

logging.basicConfig(level=logging.INFO)


def create_data_store(project_id: str, data_store_id: str, display_name: str):
    """
    Create a data store in the default collection of the project.
    Args:
        project_id: The project ID of the project to create the data store in.
        data_store_id: The ID of the data store to create.
        display_name: The display name of the data store to create.
    Returns:
        None
    e.g.:
    from dt_gen_ai_hackathon_helper.vertex_ai_search import vertex_ai_search

    display_name = "data_store_test_from_notebook"
    data_store_name = "alphabet_investor_pdfs"
    hackathon_team_name = "team_1"

    vertex_ai_search.create_data_store(PROJECT_ID, data_store_name + "_" + hackathon_team_name, display_name)
    """
    # Get the access token from gcloud
    process = subprocess.Popen(["gcloud", "auth", "print-access-token"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    access_token = out.decode('utf-8').strip()

    # Define headers and payload
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Goog-User-Project": project_id,
        "Content-Type": "application/json"
    }

    payload = {
        "displayName": display_name,
        "industryVertical": "GENERIC",
        "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
        "contentConfig": "CONTENT_REQUIRED",
        "searchTier": "STANDARD",
        "searchAddOns": ["LLM"]
    }

    parent = f"projects/{project_id}/locations/global/collections/default_collection"
    url = f"https://discoveryengine.googleapis.com/v1alpha/{parent}/dataStores/{data_store_id}"
    # Make the request
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200 or response.status_code == 201:
        logging.info("Data store created successfully.")
        logging.info(json.dumps(response.json(), indent=4))
    else:
        logging.error(f"Failed to create data store. Status code: {response.status_code}")
        logging.error(response.text)


def ingest_documents(
        project_id: str,
        location: str,
        data_store_id: str,
        gcs_uri: Optional[str] = None,
        bigquery_dataset: Optional[str] = None,
        bigquery_table: Optional[str] = None,
        data_schema: str = "content",
) -> str:
    """
    Ingest documents into a data store.
    Args:
        project_id: The project ID of the project to import documents into.
        location: The location of the data store to import documents into.
        data_store_id: The ID of the data store to import documents into.
        gcs_uri: The GCS URI of the documents to import.
        bigquery_dataset: The BigQuery dataset of the documents to import.
        bigquery_table: The BigQuery table of the documents to import.
        data_schema: The data schema of the documents to import.
    Returns:
        The operation name of the import documents operation.
    """
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # The full resource name of the search engine branch.
    # e.g. projects/{project}/locations/{location}/dataStores/{data_store_id}/branches/{branch}
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )

    if gcs_uri:
        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            gcs_source=discoveryengine.GcsSource(
                input_uris=[gcs_uri], data_schema=data_schema
            ),
            # Options: `FULL`, `INCREMENTAL`
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )
    else:
        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            bigquery_source=discoveryengine.BigQuerySource(
                project_id=project_id,
                dataset_id=bigquery_dataset,
                table_id=bigquery_table,
                data_schema="custom",
            ),
            # Options: `FULL`, `INCREMENTAL`
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
        )

    # Make the request
    operation = client.import_documents(request=request)

    logging.info("Waiting for document import to complete...")
    response = operation.result()

    # Once the operation is complete,
    # get information from operation metadata
    metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)

    # Handle the response
    logging.info(f"Result: {response}")

    return operation.operation.name


def create_search_ai_app(project_id: str, display_name: str, data_store_id: str, solution_type_search: str):
    """
    Create a search AI app.
    Args:
        project_id: The project ID of the project to create the search AI app in.
        display_name: The display name of the search AI app to create.
        data_store_id: The ID of the data store to create the search AI app in.
        solution_type_search: The solution type of the search AI app to create.
    Returns:
        None
    e.g.:
    from dt_gen_ai_hackathon_helper.vertex_ai_search import vertex_ai_search

    display_name = "app_team_1"
    solution_type_search = "SOLUTION_TYPE_SEARCH"

    vertex_ai_search.create_search_ai_app(PROJECT_ID, display_name, data_store_id, solution_type_search)
    """
    # Get the access token from gcloud
    process = subprocess.Popen(["gcloud", "auth", "print-access-token"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    access_token = out.decode('utf-8').strip()

    # Define headers and payload
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Goog-User-Project": project_id,
    }

    payload = {
        "displayName": display_name,
        "dataStoreIds": [data_store_id],
        "solutionType": [solution_type_search]
    }

    url = f"https://discoveryengine.googleapis.com/v1alpha/projects/{project_id}/locations/global/collections/default_collection/engines?engineId={data_store_id}"

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)
        logging.info(f"Success: {response.json()}")
    except requests.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logging.error(f"An error occurred: {err}")
