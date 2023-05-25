# Generative AI LangChain Hackathon

LangChain is a Python framework for developing applications using language models. It abstracts the connection between applications and LLMs, allowing a loose coupling between code and specific foundation models like Google PaLM.

## Workshop
Open `hackathon.ipynb` in JuPyter labs or as a Google CoLab notebook. The notebook is self-contained (includes python `pip` install command), but does assume that `wget` is available as a command line tool.

## Going further
After completing the workshop, an example setup for deploying the knowledge worker to production is viewable in `/gen_ai_hackathon`. The next steps covered include separating the GradI/O frontend into a separate server, and creating a FastAPI LangChain API for serving requests. Example CI:CD config files, such as `Dockerfile` and `cloudbuild.yaml` are also included to demonstrate how the servers may be built as Docker containers and stored in a Google Cloud Artifact Registry for deployment with Cloud Run.

## Running the notebook locally
To run the notebook in a `poetry` managed environment:

1. Install the virtual environment (set `virtualenvs.in-project` as `true` to create the `.venv` environment in the project folder).
```bash
poetry config virtualenvs.in-project true
poetry install
```

2. Set the kernel interpreter as `.venv/bin/python`, or start a JuPyter server using `poetry run jupyter notebook bias_generator.ipynb`.
