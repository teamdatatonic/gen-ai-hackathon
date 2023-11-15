dev: ## run streamlit locally
	@poetry run python main.py

install: ## setup and install poetry dependencies
	@poetry config virtualenvs.in-project true; poetry install

help: ## display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
