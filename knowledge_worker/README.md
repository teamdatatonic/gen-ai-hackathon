<h1 align="center"> Generative AI Hackathon</h1>
<table align="center">
    <td>
        <a href="https://colab.research.google.com/github/teamdatatonic/gen-ai-hackathon/blob/main/knowledge_worker/hackathon.ipynb">
            <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo">
            <span style="vertical-align: middle;">Run in Colab</span>
        </a>
    </td>
    <td>
        <a href="https://github.com/teamdatatonic/gen-ai-hackathon/blob/main/knowledge_worker/hackathon.ipynb">
            <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
            <span style="vertical-align: middle;">View on GitHub</span>
        </a>
    </td>
    <td>
        <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/teamdatatonic/gen-ai-hackathon/main/knowledge_worker/hackathon.ipynb">
            <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"> 
            <span style="vertical-align: middle;">Open in Vertex AI Workbench</span>
        </a>
    </td>
</table>
<hr>

**➡️ Your task:** Learn about Generative AI by building your own Knowledge Worker using Python and LangChain!

**❗ Note:** This workshop has been designed to be run in Google CoLab and Jupyter Notebook. Support for running the workshop locally or using VertexAI Workbench is provided, but we heavily recommend CoLab for the best experience.

The notebook is self-contained (includes python `pip` install commands), however, the following pre-requisites are required to get started:
- Google Cloud Project with  Vertex AI API enabled.
- A google account with access to the needed resources (see below).

## Going further

After completing the workshop, an example setup for deploying the knowledge worker to production is viewable in [gen_ai_hackathon](gen_ai_hackathon). 
The next steps covered include separating the Gradio front-end into a separate server, and creating a FastAPI LangChain API for serving requests. 

## Running the notebook for a hackathon event

1. Create a dedicated Google Cloud project with Vertex AI enabled.
2. Add each user with their own Google account with the following IAM roles:
    - `Vertex AI User` (roles/aiplatform.user) :for vertex ai endpoints
    - `Storage Object Viewer` (roles/storage.objectViewer) :to download demo and webarchive materials
    - `Service Usage Viewer` (roles/serviceusage.serviceUsageViewer) :to read the contents of buckets for dynamic selection
    - `Discovery Engine Admin` (roles/discoveryengine.admin): to create and edit applications and data stores
3. Confirm that users can access the GCP resources.
4. ❗ Post-workshop, remember to delete all the users from the project.
