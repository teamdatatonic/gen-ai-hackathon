# Generative AI Hackathon

<a target="_blank" href="https://colab.research.google.com/github/teamdatatonic/gen-ai-hackathon/blob/main/hackathon.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

LangChain is a Python framework for developing applications using language models. 
It abstracts the connection between applications and LLMs, allowing a loose coupling between code and specific foundation models like Google PaLM.

## The challenge

Open [the notebook](hackathon.ipynb) in Jupyter or [Google CoLab](https://colab.research.google.com/github/teamdatatonic/gen-ai-hackathon/blob/main/hackathon.ipynb). 
The notebook is self-contained (includes python `pip` install commands).
However, the following pre-requisites are required to get started:

- Google Cloud Project with  Vertex AI API enabled

## Going further

After completing the workshop, an example setup for deploying the knowledge worker to production is viewable in [gen_ai_hackathon](gen_ai_hackathon). 
The next steps covered include separating the Gradio front-end into a separate server, and creating a FastAPI LangChain API for serving requests. 

## Running the notebook for a Hack event

1. Create a dedicated Google Cloud project with Vertex AI enabled.
2. Create a service account with access to Vertex AI (Vertex AI user).
3. Distribute the JSON credentials for this service account, to allow participants to impersonate the SA and authenticate to access the Vertex AI endpoint.
4. Post-workshop, remember to delete the key to maintain security and prevent further billing.