<h1 align="center"> Generative AI Hackathon: Analytics Assistant</h1>
<table align="center">
</table>
<hr>

**➡️ Your task:** Learn about Generative AI by building your own Analytics Assistant using Python and LangChain!

**❗ Note:** This workshop has been designed to be run in Vertex Workbench

- Instructions on how launch the notebook using Vertex Workbench are provided in each notebook.

The notebook is self-contained (includes python `pip` install commands), however, the following pre-requisites are required to get started:
- Google Cloud Project with Vertex AI API enabled.
- A `credentials.json` file for accessing the Vertex AI API via a service account.


## Running the notebook for a hackathon event

1. Create a dedicated Google Cloud project with Vertex AI enabled.
2. Create a service account roles: `Vertex AI User` (for vertex ai endpoints), `Biq Query Admin` (to query your databse)
3. Distribute the JSON credentials for this service account, to allow participants to impersonate the SA and authenticate to access the Vertex AI endpoint.
4. ❗ Post-workshop, remember to delete the key to maintain security and prevent further billing.
