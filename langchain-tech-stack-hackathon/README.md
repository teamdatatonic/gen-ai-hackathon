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
- [DirEnv](https://direnv.net/) (or another terminal secret manager).

## Task 1 walkthrough

0. Either clone/fork this repository locally, or copy the following files into a blank project: `pyproject.toml`, `poetry.lock`, `.gitignore`.

1. Create a `.envrc` file and populate it with this template:

```bash
export GOOGLE_CLOUD_PROJECT_ID=dt-gen-ai-hackathon-dev
export DATA_STORE_ID=<your-data-store-id> # replace with your data store ID for this workshop
export MODEL_TYPE=chat-bison@001
```

_❗ If you don't want to use DirEnv, just run each line as a separate terminal command - the environment variables will persist till you close the terminal window_

2. Open a terminal at the root of your project and run these commands:

```sh
pyenv shell 3.11.6
direnv allow # if using DirEnv as a secret manager
poetry install
source ./.venv/bin/activate
poetry run langchain app new api --package rag-google-cloud-vertexai-search
```

_❗ Type 'Y' when prompted to install rag-google-cloud-vertexai-search as a mirrored module_

- This creates a new Poetry environment with the LangChain CLI tool installed.
- Then activates the virtual environment to access the tool.
- And finally create a new LangServe app, using the [rag-google-cloud-vertexai-search](https://github.com/langchain-ai/langchain/tree/master/templates/rag-google-cloud-vertexai-search) component (written by our own Juan Calvo!).

3. Replace the `add_routes(app, NotImplemented)` route in `api/app/server.py` with a route for your component chain:

```python
from rag_google_cloud_vertexai_search.chain import chain as rag_google_cloud_vertexai_search_chain

add_routes(app, rag_google_cloud_vertexai_search_chain, path="/rag-google-cloud-vertexai-search")
```

4. In your terminal, run these commands to start the LangServe server:

```sh
cd api
poetry run langchain serve
```

6. Visit http://127.0.0.1:8000/rag-google-cloud-vertexai-search/playground/ in your web browser and experiment with the API.

REPLACE_WITH_VIDEO

## Task 2 walkthrough

Let's add LangSmith logging to your project.

0. Create a LangSmith account at https://smith.langchain.com/, then use our partner key to skip the waitlist (shared with Hackathon participants prior to this workshop, ask your workshop lead if you haven't recieved a code).

1. Update your `.envrc` file with these additional environment variables. Find your API key on the LangSmith platform (follow the video below):

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-langsmith-api-key>
export LANGCHAIN_PROJECT=rag-google-cloud-vertexai-search  # if not specified, defaults to "default"
```

_❗ Update `LANGCHAIN_API_KEY` with your API key_

REPLACE_WITH_VIDEO

2. Restart your LangServe server (`ctrl-c` in the terminal to stop the process, then re-run it with these commands:)

```sh
direnv allow # allow your terminal to access your new `.envrc` secrets
poetry run langchain serve
```

3. Use the playground to test your application a few times - this usage is now being logged into LangSmith.

4. Open LangSmith (https://smith.langchain.com/) and visit the project page. View one of your traces (created when you tested the playground demo) and use it to create a dataset (name it `vertex-ai-search-dataset`).

REPLACE_WITH_VIDEO

5. We can automate testing, using our dataset as examples. Create a new PyTest file `test_chain.py` in `api/packages/rag-google-cloud-vertexai-search/tests/`:

```python
from rag_google_cloud_vertexai_search.chain import chain
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

6. Stop your LangServe server (`ctrl+c`) and use the following command to run the new test:

```sh
poetry run python -m pytest -s .
```

- The `pytest -s .` command searches the repository from the current folder, and finds all tests in any sub-folders.

7. View your dataset test runs, and add the trace to a new annotation queue (name it `vertex-ai-search-annotations`).

REPLACE_WITH_VIDEO

8. View your annotation queue and explore the review interface.

REPLACE_WITH_VIDEO

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
