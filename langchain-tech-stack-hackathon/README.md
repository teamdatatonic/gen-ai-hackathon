<h1 align="center"> LangChain Tech-Stack Hackathon</h1>
<table align="center">
    <td>
        <a href="https://github.com/teamdatatonic/gen-ai-hackathon/blob/main/langchain-tech-stack-hackathon/">
            <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
            <span style="vertical-align: middle;">View on GitHub</span>
        </a>
    </td>
</table>
<hr>

**➡️ Your task:** Learn the complete LangChain tech stack by building your own Knowledge Worker using Python and LangChain!

**❗ Note:** This workshop has been designed to be run locally in an IDE like VSCode.

The following pre-requisites are required to get started:

- Google Cloud Project with Vertex AI API enabled.
- A Google account with access to the needed resources ([see below](#running-a-hackathon-event)).
- The `gcloud` CLI tool, configured to access the hackathon GCP.
- Python (this tutorial assumed 3.11, but other versions will work) (use [PyEnv](https://github.com/pyenv/pyenv) for version management).
- Poetry (^1.6.1)
- A [LangSmith](https://smith.langchain.com/) account (this will require sign-up - speak to @zacharysmithdatatonic for a sign-up code to gain access immediately).
- [DirEnv](https://direnv.net/) (or another terminal secret manager).
- A blank project, open in VSCode.

## Tutorial walkthrough

0. If using a secret file (e.g.: `.envrc`), create a `.gitignore` file [like this one](https://www.toptal.com/developers/gitignore/api/direnv,python,visualstudiocode,macos), to prevent accidentally sharing your API keys.

1. Create a `.envrc` file and populate it with this template (adding in your LangSmith API key):

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-langsmith-api-key>
export LANGCHAIN_PROJECT=rag-google-cloud-vertexai-search  # if not specified, defaults to "default"

export GOOGLE_CLOUD_PROJECT_ID=dt-gen-ai-hackathon-dev
export DATA_STORE_ID=<your-data-store-id>
export MODEL_TYPE=chat-bison@001
```

2. Open a new terminal and run these commands:

```bash
pyenv shell 3.11.6
poetry init -n --python=3.11.6 && poetry add langchain-cli
source ./.venv/bin/activate
poetry run langchain app new api --package rag-google-cloud-vertexai-search
```

_❗ Type 'Y' when prompted to install rag-google-cloud-vertexai-search as a mirrored module_

- This creates a new Poetry environment with the LangChain CLI tool installed.
- Then activates the virtual environment to access the tool.
- And finally create a new LangServe app, using the [rag-google-cloud-vertexai-search](https://github.com/langchain-ai/langchain/tree/master/templates/rag-google-cloud-vertexai-search) component (written by our own Juan Calvo!).

3. Replace the `add_routes(app, NotImplemented)` route in `api/app/server.py` with a route for your component chain:

```python
from rag_google_cloud_vertexai_search import chain as rag_google_cloud_vertexai_search_chain

add_routes(app, rag_google_cloud_vertexai_search_chain, path="/vertex-ai-search")
```

4. Update the project `name` in `api/pyproject.toml`:
```toml
name = "api"
```

_❗ This can be any name other than `__app_name__`_

5. Open a new terminal in `/api` and run these commands:

```bash
pyenv shell 3.11.6
poetry install
source ./.venv/bin/activate
poetry add google-cloud-discoveryengine
poetry run langchain serve
```

- We want to close our initial terminal in order to close the first virtual environment from step 2, to then enable and activate the API Poetry environment for serving.

6. Visit http://127.0.0.1:8000/vertex-ai-search/playground/ in your web browser and play with the chain.

7. Open LangSmith (https://smith.langchain.com/) and visit the project page. View one of your traces (created when you tested the playground demo) and use it to create a dataset (name it `vertex-ai-search-dataset`).

8. View the dataset and click `New Test Run` to get a code snippet:

```python
client = langsmith.Client()
chain_results = client.run_on_dataset(
	dataset_name="vertex-ai-search-dataset", # this will change if you use a different dataset name.
	llm_or_chain_factory=chain,
	project_name="...", # this will be a random string
	concurrency_level=5,
	verbose=True,
)
```

9. Create a new PyTest file `test_chain.py` in `api/packages/rag-google-cloud-vertexai-search/tests/`:

```python
from pirate_assistant.chain import chain
import langsmith
from datetime import datetime # import datetime module to get a timestamp

def test_chain():
	client = langsmith.Client()

	chain_results = client.run_on_dataset(
		dataset_name="vertex-ai-search-dataset",
		llm_or_chain_factory=chain,
		project_name=f"vertex-ai-search-dataset-test-{int(datetime.now().strftime('%Y%m%d%H%M%S'))}", # use a timestamped unique project name each re-run
		concurrency_level=5,
		verbose=True,
	)
```

- We update the `project_name` from the LangSmith code snippet default ➡️ a timestamped unique string because each dataset test run must have a unique name (the LangSmith code snippet is single-use only, so we need to fix this).

10. Open a terminal in `/api` and run these commands:

```bash
poetry add pytest --group=dev
poetry run python -m pytest -s .
```

- We add `pytest` to the `dev` group since we'll only be running tests during development, not once we move the code to production.
- The `pytest -s .` command searches the repository from the current folder, and finds all tests in any sub-folders.

11. View your dataset test runs, and add the trace to a new annotation queue (name it `vertex-ai-search-annotations`).

12. View your annotation queue and explore the review interface.

🎉🎉 **Congratulations!** 🎉🎉
You've completed this tutorial and now have a complete LangChain project performing RAG with Vertex AI Search.

## Running a hackathon event

1. Create a dedicated Google Cloud project with Vertex AI enabled.
2. Create a Vertex AI Search app:
    - Create an app [here](https://cloud.google.com/generative-ai-app-builder/docs/create-engine-es).
    - Create a data store alongside the app (or separately [here](https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es)).
        - A suitable dataset to test this template with is the Alphabet Earnings Reports, which you can find [here](https://abc.xyz/investor/). The data is also available at gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs.
3. Add each user with their own Google account with the following IAM roles:
    - `Vertex AI User` (roles/aiplatform.user): for vertex ai endpoints
4. Confirm that users can access the GCP resources.
5. ❗ Post-workshop, remember to delete all the users from the project.
