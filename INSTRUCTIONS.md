# Intro to LangChain

LangChain is a Python framework for developing applications using language models. It abstracts the connection between applications and LLMs, allowing a loose coupling between code and specific providers like OpenAI, Anthropic, etc.

This guide details how to get started with LangChain, and walks-through setting up a Q&A over Documentation example.

## Prerequisites

### `pyproject.toml`

```TOML
[tool.poetry]
name = "langchain-demo"
version = "0.1.0"
description = ""
authors = ["YOUR NAME <YOUR.EMAIL@datatonic.com>"]
license = "MIT"
readme = "README.md"
packages = [{include = "langchain_demo"}]

[tool.poetry.dependencies]
python = "^3.11"
langchain = "^0.0.150"
openai = "^0.27.4"
chromadb = "^0.3.21"
tiktoken = "^0.3.3"
beautifulsoup4 = "^4.12.2"
gradio = "^3.28.2"
unstructured = "^0.6.3"
tqdm = "^4.65.0"

[tool.poetry.group.dev.dependencies]
jupyter = "^1.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.scripts]
langchain_demo = "langchain_demo.qna:main"
```

Customise the file as appropriate - the authors section should be set, and the python version dependency can be set as low as ^3.9 whilst maintaining compatibility.

### Setup the project using Poetry

Create a local project folder containing your `pyproject.toml` file, then install dependencies:

```bash
poetry install
```

This guide assumes the following project layout:

```
.
├── [pyproject.toml](#pyprojecttoml)
├── [.gitignore](#gitignore)
├── [.envrc](#envrc)
└── langchain_demo
    ├── __init__.py
    └── [qna.py](#qnapy)
```
Create empty files for this folder structure above, then populate them with the content from this guide.

### `.gitignore`

Use this `.gitignore` file if you’re tracking this project on GitHub, to prevent secrets from the `.envrc` file from being made public.

[_gitignore_](https://www.toptal.com/developers/gitignore/api/python,direnv,visualstudiocode)

Also add `chromadb` and (e.g.) `datatonic.com` the the `.gitignore`: the vector database should be stored on GCS instead of the GitHub repository, and the target webpage / HTML files can be discarded once embeddings have been generated.

### `.envrc`

```bash
export OPENAI_API_KEY=""
```
We use OpenAI for our embeddings model and LLM, but this can be changed easily.

## Q&A over Documentation

Creating a custom knowledge worker is the “Hello World!” of LLMOps. Loading documents, creating embeddings, storing in a vector database and using an LLM to answer queries with knowledge from that database can be achieved in a few lines of Python.

First, we need to select our documentation. LangChain supports [numerous methods](https://python.langchain.com/en/latest/modules/indexes/document_loaders.html) for loading documents. As we will be constructing our knowledge worker using website data, we first download the files locally using `wget`:

```bash
wget -r -A.html -P rtdocs https://gcloud.readthedocs.io/en/latest/
```

Note how filepaths to the `.html` files form proper URL paths, we'll use this later to reference our answers with proper hyperlinks.

The downloaded website can then be loaded as documents using the LangChain [`DirectoryLoader`](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/directory_loader.html) and [`UnstructuredHTMLLoader`](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/html.html) loaders.

### `qna.py`

```python
import os
import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Set the environment variables for persisting our vectorstore
PERSIST_DIR = "chromadb"
SOURCE_DIR = "datatonic.com"
```

```python
def load_documents():
    # Load the documentation using a HTML parser
    loader = DirectoryLoader(SOURCE_DIR, glob="**/*.html", loader_cls=UnstructuredHTMLLoader, show_progress=True)
    documents = loader.load()

    # Individual documents will often exceed the 4096 token limit for GPT-3.
    # By splitting documents into chunks of 1000 token
    # These chunks fit into the token limit alongside the user prompt
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    return texts



def embed_documents(texts):
    # We use OpenAI embeddings model, however other models can be substituted here
    embeddings = OpenAIEmbeddings()
    # We create a vector store database relating documents to embeddings
    # This embedding database is used to relate user queries to relevant documentation
    vector_store = Chroma.from_documents(
        persist_directory=PERSIST_DIR,
        documents=texts,
        embedding=embeddings,
    )

    vector_store.persist()

    return vector_store


def load_embeddings():
    # We use OpenAI embeddings model, however other models can be substituted here
    embeddings = OpenAIEmbeddings()

    # Creating embeddings with each re-run is highly inefficient and costly.
    # We instead aim to embed once, then load these embeddings from storage.
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    return vector_store
```

```python
def qa_with_sources_chain(k: int, temperature: float) -> ConversationalRetrievalChain:
    # If the vectorstore embedding database has been created, load it
    if os.path.isdir(PERSIST_DIR):
        print(f"Loading {PERSIST_DIR} as vector store")
        vector_store = load_embeddings()
    # If it exists, create it
    else:
        print(f"Creating new vector store in dir {PERSIST_DIR}")
        texts = load_documents()
        vector_store = embed_documents(texts)

    # A vector store retriever relates queries to embedded documents
    retriever = vector_store.as_retriever(k=k)
    # The selected OpenAI model uses embedded documents related to the query
    # It parses these documents in order to answer the user question.
    # We use the OpenAI LLM, however other models can be substituted here
    model = OpenAI(temperature=temperature)

    # A conversation retrieval chain keeps a history of Q&A / conversation
    # This allows for contextual questions such as "give an example of that (previous response)".
    # The chain is also set to return the source documents used in generating an output
    # This allows for explainability behind model output.
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
    )

```

```python
def main():
    # The 'k' value indicates the number of sources to use per query.
    # 'k' as in 'k-nearest-neighbours' to the query in the embedding space.
    # 'temperature' is the degree of randomness introduced into the LLM response.
    qa_chain = qa_with_sources_chain(k=2, temperature=0.0)

    def q_a(question: str, history: list):
        # map history (list of lists) to expected format of chat_history (list of tuples)
        chat_history = map(tuple, history)
        # Engineer the prompt to respond in markdown format, to display nicely in the chat log
        prompt = f"Respond to the following sentence using markdown format, wrap all code examples in ``. {question}"
        # Query the LLM to get a response
        # First the Q&A chain will collect documents semantically similar to the question
        # Then it will ask the LLM to use this data to answer the user question
        # We also provide chat history as further context
        response = qa_chain(
            {
                "question": prompt,
                "chat_history": chat_history,
            }
        )

        # Format source documents (sources of excerpts passed to the LLM) into links the user can validate
        sources = [
            "[{0}]({0})".format(doc.metadata["source"])
            for doc in response["source_documents"]
        ]

        # Return the LLM answer, and list of sources used (formatted as a string)
        return response["answer"], "\n\n".join(sources)

    # Build a simple GradIO app that accepts user input and queries the LLM
    # Then displays the response in a ChatBot interface, with markdown support.
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        def submit(msg, chatbot):
            # First create a new entry in the conversation log
            msg, chatbot = user(msg, chatbot)
            # Then get the chatbot response to the user question
            chatbot = bot(chatbot)
            return msg, chatbot

        def user(user_message, history):
            # Return "" to clear the user input, and add the user question to the conversation history
            return "", history + [[user_message, None]]

        def bot(history):
            # Get the user question from conversation history
            user_message = history[-1][0]
            # Get the response and sources used to answer the user question
            bot_message, bot_sources = q_a(user_message, history[:-1])

            # Using a template, format the response and sources together
            bot_template = "{0}\n\n<details><summary><b>Sources</b></summary>\n\n{1}</details>"
            # Place the response into the conversation history and return
            history[-1][1] = bot_template.format(
                bot_message, bot_sources
            )
            return history

        # Set a page title
        gr.Markdown("# Custom knowledge worker")
        # Create a chatbot conversation log
        chatbot = gr.Chatbot(label="🤖 knowledge worker")
        # Create a textbox for user questions
        msg = gr.Textbox(label="👩‍💻 user input", info="Query information from the custom knowledge base.")

        # Align both buttons on the same row
        with gr.Row():
            send = gr.Button(value="Send", variant="primary").style(size="sm")
            clear = gr.Button(value="Clear History", variant="secondary").style(size="sm")

        # Submit message on <enter> or clicking "Send" button
        msg.submit(submit, [msg, chatbot], [msg, chatbot], queue=False)
        send.click(submit, [msg, chatbot], [msg, chatbot], queue=False)

        # Clear chatbot history on clicking "Clear History" button
        clear.click(lambda: None, None, chatbot, queue=False)

    # Create a queue system so multiple users can access the page at once
    demo.queue()
    # Launch the webserver locally
    demo.launch()

```

### Running the program

```bash
poetry run langchain_demo
```
As we are using Poetry, we can call the script from an entry-point function such as main(). This starts a local web-server for the GradIO app, which we can open in the browser.

Testing our demo, we can query our knowledge base for answers (code emphasis added). Since we used a `ConversationalRetrievalChain`, we can also correct the model when it gives the wrong response and prompt it to fix it’s mistake, or ask for further detail on a previous response.

> 👩‍💻: how to install the gcloud cli
> 
> 🦜:  You can install the gcloud CLI by running the command `$ gcloud components update`.
> 
> 👩‍💻: that command requires gcloud to be installed, how can I install gcloud initially?
> 
> 🦜:  You can install gcloud initially by running the command '`pip install google-cloud`' in your terminal.
> 
> 👩‍💻: how can I set the target project
> 
> 🦜:  You can set the target project for the gcloud CLI by using the command `$ gcloud config set project my-new-default-project`.
> 
> 👩‍💻: what is the gcloud cli?
> 
> 🦜:  The gcloud CLI is a command line interface for Google Cloud Platform services.
> 
> 👩‍💻: explain the above in more detail
> 
> 🦜:  The gcloud CLI is a tool used to authenticate and configure credentials for Google Cloud services. It can be used to change the default project ID, update components, and authenticate the CLI itself.
>