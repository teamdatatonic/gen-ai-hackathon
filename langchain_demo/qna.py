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
