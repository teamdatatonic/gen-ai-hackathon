from langchain.chains import SQLDatabaseSequentialChain
from langchain import PromptTemplate


TEST_PROMPT ='''
Run SQL code to obtain the answer to the following question:
{question}
'''

def create_db_chain(llm, db, question):
    """ Create a Q&A conversation chain using the VertexAI LLM.

    """
    
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm,
        db,
        verbose=True,
        return_intermediate_steps=True,
    )
    test_prompt = PromptTemplate(template=TEST_PROMPT, input_variables=["question"])

    output = db_chain(test_prompt.format(question=question))
    return output['result']






