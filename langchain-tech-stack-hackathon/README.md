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

**➡️ Your task:** Learn the complete LangChain tech stack by deploying a REST API using LangChain and LangServe, and learn about LLM logging and testing using the LangSmith platform.

**❗ Note:** This workshop has been designed to be run locally in an IDE like VSCode.

**🙋🏻‍♀️ Top Tip:** Read through the entire walkthrough before diving into your own implementation/code - some steps can be skipped e.g.: if you've cloned this repository.

## Task 0 (prerequisites and setup)

0. The following pre-requisites are required to get started:

- Google Cloud Project with Vertex AI API enabled.
- A Google account with access to the required resources ([see below](#running-a-hackathon-event)).
- The `gcloud` CLI tool, configured to access the hackathon GCP ([install instructions](https://cloud.google.com/sdk/docs/install)).
- Python (this tutorial assumed 3.11.6, but other versions will work) (use [PyEnv](https://github.com/pyenv/pyenv) for version management).
- Poetry (^1.6.1) ([install instructions](https://python-poetry.org/docs/#installation))
- [DirEnv](https://direnv.net/) (or another terminal secret manager).

**❗ Note:** Perform the following commands from any terminal.

1. Set your `gcloud` CLI tool to the correct project:

```sh
gcloud config set project dt-gen-ai-hackathon-dev
```

2. If using `pyenv`, install the correct version of Python:

```sh
pyenv install 3.11.6
```

## Task 1 walkthrough

0. Create a `.gitignore` file with [this content](https://www.toptal.com/developers/gitignore/api/direnv,python,visualstudiocode,macos).

1. Create a `.envrc` file and populate it with this template:

```bash
export GOOGLE_CLOUD_PROJECT_ID=dt-gen-ai-hackathon-dev
export DATA_STORE_ID=<your-data-store-id> # replace with your data store ID for this workshop
export MODEL_TYPE=chat-bison@001
```

_❗ If you don't want to use DirEnv, just run each line as a separate terminal command - the environment variables will persist till you close the terminal window_

If there is a bottleneck with concurrent hackathon users accessing the search application later in the tutorial, return to this step and create your own personal data store / search application using [this](https://github.com/langchain-ai/langchain/tree/master/templates/rag-google-cloud-vertexai-search#environment-setup) guide. Once created, update the `GOOGLE_CLOUD_PROJECT_ID` and `DATA_STORE_ID` accordingly.

2. Open a terminal at the root of your project:

- Ensure the right Python version is activated, DirEnv can access your secrets file, and set Poetry to create the `.venv` environment folder locally (this helps if you need to tidy-up/reset the environment later):

```sh
pyenv shell 3.11.6
direnv allow
poetry config virtualenvs.in-project true
```

- If you have cloned this repository, run these commands:

```sh
cd langchain-tech-stack-hackathon
poetry install
```

- _Else_, initialise Poetry from scratch and install these requirements:

```sh
poetry init -n --python=3.11.6
poetry add "langchain-cli[serve]"
poetry add google-cloud-discoveryengine
poetry add pytest --group=dev
```

- _Finally_, activate the environment and use the LangChain CLI to create a new LangServe app:

```sh
source ./.venv/bin/activate
poetry run langchain app new api --package rag-google-cloud-vertexai-search --package vertexai-chuck-norris
```

_❗ Type 'Y' when prompted to install components as mirrored modules (meaning we can develop them locally but treat them like pip-installed packages)._

- This creates a new Poetry environment with the LangChain CLI tool installed.
- Then activates the virtual environment to access the tool.
- Create a new LangServe app, named `api`.
    - Add the [rag-google-cloud-vertexai-search](https://github.com/langchain-ai/langchain/tree/master/templates/rag-google-cloud-vertexai-search) component (written by our own Juan Calvo!).
    - Add the [vertexai-chuck-norris](https://github.com/langchain-ai/langchain/tree/master/templates/vertexai-chuck-norris) component.

3. Replace the `add_routes(app, NotImplemented)` route in `api/app/server.py` with routes for your chains:

```python
from rag_google_cloud_vertexai_search.chain import chain as rag_google_cloud_vertexai_search_chain
from vertexai_chuck_norris.chain import chain as vertexai_chuck_norris_chain

add_routes(app, rag_google_cloud_vertexai_search_chain, path="/rag-google-cloud-vertexai-search")
add_routes(app, vertexai_chuck_norris_chain, path="/vertexai-chuck-norris")
```


https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/f6880855-0c72-4eba-892b-788aa8b3c253


_❗ Be sure to use the snippet above and NOT the suggested snippet generated by the LangChain CLI - they look similar but the LangChain CLI has a bug that misses the `.chain` part of the import path._

4. In your terminal, run these commands to start the LangServe server:

```sh
cd api
poetry run langchain serve
```

6. Visit http://127.0.0.1:8000/rag-google-cloud-vertexai-search/playground/ and http://127.0.0.1:8000/vertexai-chuck-norris/playground/ in your web browser and experiment with the API endpoints.



https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/08e3149e-be12-4d8c-a0d0-452fa07563c4



https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/b50b1372-3164-498e-9411-6b8580843ec8



🎉🎉 **Congratulations!** 🎉🎉
You've completed this tutorial and now have a complete LangChain API performing RAG with Vertex AI Search.

## Task 2 walkthrough

Let's add LangSmith logging to your project.

_❗ The `rag-google-cloud-vertexai-search` does not currently support dataset testing, hence the use of the `vertexai-chuck-norris` component._

0. Create a LangSmith account at https://smith.langchain.com/, then use our partner key to skip the waitlist (shared with Hackathon participants prior to this workshop, ask your workshop lead if you haven't recieved a code).

1. Update your `.envrc` file with these additional environment variables. Find your API key on the LangSmith platform (follow the video below):

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-langsmith-api-key>
export LANGCHAIN_PROJECT=hackathon-logging  # if not specified, defaults to "default"
```

_❗ Update `LANGCHAIN_API_KEY` with your API key_



https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/b5c4db22-cc35-484a-9197-cf04e4fdaed4



2. Restart your LangServe server (`ctrl-c` in the terminal to stop the process, then re-run it with these commands:)

```sh
direnv allow # allow your terminal to access your new `.envrc` secrets
poetry run langchain serve
```

3. Use the [rag-google-cloud-vertexai-search](http://127.0.0.1:8000/rag-google-cloud-vertexai-search/playground/) and [vertexai-chuck-norris](http://127.0.0.1:8000/vertexai-chuck-norris/playground/) playgrounds to test your application a few times - this usage is now being logged into LangSmith.

4. Open LangSmith (https://smith.langchain.com/) and visit the project page. View one of your traces (created when you tested the playground demo) and use it to create a dataset (name it `vertexai-chuck-norris-dataset`).



https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/3f4ac1c5-e406-463d-a37c-a7d8c26a93d6



5. We can automate testing, using our dataset as examples. Create a new PyTest file `test_chain.py` in `api/packages/vertexai-chuck-norris/tests/`:

```python
from vertexai_chuck_norris.chain import chain as vertexai_chuck_norris_chain
import langsmith
from datetime import datetime # import datetime module to get a timestamp

def test_chain():
	client = langsmith.Client()

	chain_results = client.run_on_dataset(
		dataset_name="vertexai-chuck-norris-dataset",
		llm_or_chain_factory=vertexai_chuck_norris_chain,
		project_name=f"vertexai-chuck-norris-dataset-test-{int(datetime.now().strftime('%Y%m%d%H%M%S'))}", # use a timestamped unique project name each re-run
		concurrency_level=5,
		verbose=True,
	)
```

6. Stop your LangServe server (`ctrl+c`) and use the following command to run the new test:

```sh
poetry run python -m pytest -s .
```

- The `pytest -s .` command searches the repository from the current folder, and finds all tests in any sub-folders.

7. View your dataset test run, and annotate the result.



https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/3909ec7b-4037-403d-a75e-3e30ec7462cb



8. Return to your project logging and add the `rag-google-cloud-vertexai-search` run to a new annotation queue (name it `rag-vertexai-search-annotations`). Next, view your annotation queue and explore the review interface.



https://github.com/teamdatatonic/gen-ai-hackathon/assets/125287327/4d555d57-3858-4e4c-bfb0-d894e8b8757e



🎉🎉 **Congratulations!** 🎉🎉
You've explored the LangSmith platform for monitoring and dataset testing - if you want to explore more features of the platform, take a look at the LLM-based evaluators that can be used to evaluate dataset test runs automatically, or check out the LangChain Prompt Hub to look at community-shared LLM prompt engineering examples and techniques.

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
