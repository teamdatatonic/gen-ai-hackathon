from langchain.chains import ConversationalRetrievalChain
from langchain.llms import VertexAI


def create_qa_chain(vector_store, condense_question_prompt, k=4, temperature=0.0):
    """ Create a Q&A conversation chain using the VertexAI LLM.

    Arguments:
        vector_store (object): The vectorstore containing our knowledge.
        condense_question_prompt (PromptTemplate): The prompt template used to prompt engineer our LLM to respond in a certain tone, etc.
        k (int): the 'k' value indicates the number of sources to use per query. 'k' as in 'k-nearest-neighbours' to the query in the embedding space.
        temperature (float): the degree of randomness introduced into the LLM response.
    """

    # A vector store retriever relates queries to embedded documents
    retriever = vector_store.as_retriever(k=k)

    # The selected Google model uses embedded documents related to the query
    # It parses these documents in order to answer the user question.
    # We use the VertexAI LLM, however other models can be substituted here
    model = VertexAI(model_name='text-bison@001',temperature=temperature)

    # A conversation retrieval chain keeps a history of Q&A / conversation
    # This allows for contextual questions such as "give an example of that (previous response)".
    # The chain is also set to return the source documents used in generating an output
    # This allows for explainability behind model output.
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
        condense_question_prompt=condense_question_prompt,
    )

    return chain