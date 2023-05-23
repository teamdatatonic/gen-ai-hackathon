from googlellm import GoogleLLM, GooglePalmEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

PERSIST_DIR = "chromadb"
template = """\
You are a helpful chatbot designed to perform Q&A on a set of documents.
Always respond to users with friendly and helpful messages.
Your goal is to answer user questions using relevant sources.

You were developed by Datatonic, and are powered by Google's PaLM-2 model.

In addition to your implicit model world knowledge, you have access to the following data sources:
- Company documentation.

If a user query is too vague, ask for more information.
If insufficient information exists to answer a query, respond with "I don't know".
NEVER make up information.

Chat History:
{chat_history}
Question: {question}
"""

# The PromptTemplate reads input variables (i.e.: 'chat_history', 'question') from the template
SYSTEM_PROMPT = PromptTemplate.from_template(template)



def load_documents(source_dir):
    # Load the documentation using a HTML parser
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.html",
        loader_cls=UnstructuredHTMLLoader,
        show_progress=True,
    )
    documents = loader.load()

    return documents


def create_embeddings(source_dir):
    vector_store = load_embeddings()

    documents = load_documents(source_dir=source_dir)
    vector_store = embed_documents(vector_store, documents)

    return vector_store


def embed_documents(vector_store, documents):
    # Individual documents will often exceed the token limit.
    # By splitting documents into chunks of 1000 token
    # These chunks fit into the token limit alongside the user prompt
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Add our documents (split into shards) to the DB
    # They will be embedded using the defined GooglePalmEmbeddings model
    vector_store.add_texts(texts)

    # Persist the ChromaDB locally, so we can reload the script without expensively re-embedding the database
    vector_store.persist()

    return vector_store


def load_embeddings():
    # We use GoogleLLM embeddings model, however other models can be substituted here
    embeddings = GooglePalmEmbeddings()()

    # Creating embeddings with each re-run is highly inefficient and costly.
    # We instead aim to embed once, then load these embeddings from storage.
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    return vector_store


def qa_with_sources_chain(temperature, k):
    vector_store = load_embeddings()

    # A vector store retriever relates queries to embedded documents
    retriever = vector_store.as_retriever(k=k)

    # The selected GoogleLLM model uses embedded documents related to the query
    # It parses these documents in order to answer the user question.
    # We use the GoogleLLM LLM, however other models can be substituted here
    model = GoogleLLM(temperature=temperature)

    # A conversation retrieval chain keeps a history of Q&A / conversation
    # This allows for contextual questions such as "give an example of that (previous response)".
    # The chain is also set to return the source documents used in generating an output
    # This allows for explainability behind model output.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
        condense_question_prompt=SYSTEM_PROMPT,
    )

    return qa_chain