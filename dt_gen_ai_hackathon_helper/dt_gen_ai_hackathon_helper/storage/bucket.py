from google.cloud import storage


def list_bucket_files(bucket_name):
    # Instantiate a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all the files in the bucket
    files = bucket.list_blobs()

    # Iterate over the files and print their names
    file_names = []
    for file in files:
        file_names.append(file.name.replace(".tar.gz", ""))

    return file_names
