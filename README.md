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

## Running the notebook locally

To run the notebook in a `poetry` managed environment:

1. Install the virtual environment (set `virtualenvs.in-project` as `true` to create the `.venv` environment in the project folder).
```bash
poetry config virtualenvs.in-project true
poetry install
```

2. Set the kernel interpreter as `.venv/bin/python`, or start a JuPyter server using `poetry run jupyter notebook hackathon.ipynb`.
