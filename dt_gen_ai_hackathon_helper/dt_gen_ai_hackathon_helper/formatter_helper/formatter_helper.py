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
