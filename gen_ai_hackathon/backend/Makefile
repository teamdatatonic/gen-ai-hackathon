dev: ## run FastAPI locally
	@poetry run uvicorn main:app --reload --port=8000

test: ## run pytest locally
	@poetry run pytest

install: ## setup and install poetry dependencies
	@poetry install

help: ## display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## build docker container using cloud build (automatically pushes to artifact registry)
	@gcloud builds submit --config cloudbuild.yaml --region europe-west2 .
