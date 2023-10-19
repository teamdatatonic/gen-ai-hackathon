from typing import Dict

import pandas as pd


def format_results(results: Dict) -> pd.DataFrame:
    """
    Formats and prints the results of a question-answering query.
    Args:
        results: The results of a question-answering query.
    Returns:
        None
    """
    # Prepare the header for the table
    print("*" * 79)
    print(f"Answer: {results['result']}")
    print(f"Used {len(results['source_documents'])} relevant documents.")

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['Document', 'Source', 'First 100 characters'])

    # Loop through source documents to populate DataFrame
    counter = 1
    for doc in results["source_documents"]:
        new_row = pd.DataFrame({
            'Document': f"Dcoument {counter}",
            'Source': [gsutil_uri_to_gcs_url(doc.metadata['source'])],
            'First 100 characters': [doc.page_content[:100]]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        counter += 1

    # Print the DataFrame
    print(df)

    # Print the separator
    print("*" * 79)

    # Continue with the original printing logic
    counter = 1
    for doc in results["source_documents"]:
        print("-" * 79)
        print(f"Document {counter}")
        print("-" * 79)
        print(f"Source of content: {gsutil_uri_to_gcs_url(doc.metadata['source'])}")
        print("-" * 79)
        print(doc.page_content)
        counter += 1
    return df


def gsutil_uri_to_gcs_url(gsutil_uri: str) -> str:
    """Converts a gsutil URI to a GCS URL.

    Args:
        gsutil_uri (str): The gsutil URI to be converted, e.g., "gs://bucket-name/object-name".

    Returns:
        str: The converted GCS URL, e.g., "https://storage.googleapis.com/bucket-name/object-name".
    """
    return gsutil_uri.replace("gs://", "https://storage.googleapis.com/")
