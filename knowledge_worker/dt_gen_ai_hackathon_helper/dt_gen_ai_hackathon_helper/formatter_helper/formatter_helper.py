from typing import Dict


def format_results(results: Dict) -> None:
    """
    Formats and prints the results of a question-answering query.
    Args:
        results: The results of a question-answering query.
    Returns:
        None
    """
    counter = 1
    print("*" * 79)
    print(f"Answer: {results['result']}")
    print(f"Used {len(results['source_documents'])} relevant documents.")
    print("*" * 79)
    for doc in results["source_documents"]:
        print("-" * 79)
        print(f"Document {counter}")
        print("-" * 79)
        print(f"Source of content: {doc.metadata['source']}")
        print("-" * 79)
        print(doc.page_content)
        counter += 1


def gsutil_uri_to_gcs_url(gsutil_uri: str) -> str:
    """Converts a gsutil URI to a GCS URL.

    Args:
        gsutil_uri (str): The gsutil URI to be converted, e.g., "gs://bucket-name/object-name".

    Returns:
        str: The converted GCS URL, e.g., "https://storage.googleapis.com/bucket-name/object-name".
    """
    return gsutil_uri.replace("gs://", "https://storage.googleapis.com/")

