import ipywidgets as widgets
from dt_gen_ai_hackathon_helper.storage.bucket import list_bucket_files

def gcp_bucket_dropdown(bucket_name, label):
    # Call the function to list files in the bucket
    file_list = list_bucket_files(bucket_name)

    dropdown = widgets.Dropdown(
        options=file_list,
        value=file_list[0],
        description=label,
    )

    return dropdown

def multiline_prompt_input(initial_value):
    areabox = widgets.Textarea(value=initial_value)

    return areabox