from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import(create_pandas_dataframe_agent)
from langchain.chains import SQLDatabaseSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent 
from langchain import LLMChain,PromptTemplate
from datetime import date
from pathlib import Path
import os




def load_prompt_template(path, input_variables):
    """ Load prompt template.

    Arguments:
        path: path to promopt file
        Input_variables: Input keys defined in the prompt template.

    Returns:
        prompt template: A template defined with input variables 
    """
    with open(str(path), "r") as fp:
        return PromptTemplate(template=fp.read(), input_variables=input_variables)


prompts = Path(os.path.realpath(__file__)).parent.parent / "prompts"
prompt_generate_sql = load_prompt_template(
    prompts / "google_sql.txt", ["question", "table_info", "top_k", "today_date"]
)
prompt_resummarise = load_prompt_template(
    prompts / "resummarise.txt",
    ["query", "initial_summary", "initial_query"],
)

table_names = "customers,orders,employees,inventory,product_reviews,supplier_orders,financial_goals,marketing_campaign_analytics"



def create_db_chain(llm, db, question):
    """ Create a database query chain using the VertexAI LLM.

    Arguments:
        question: The question that is fed into SQL Chain
        LLM: The large language model.
        Database (object): The database containing our knowledge 

    Returns:
        output: Answer returned the LLM chain 
    """
   
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm,
        db,
        verbose=True,
        return_intermediate_steps=True,
    )
 
    final_prompt = prompt_generate_sql.format(
        question=question,
        table_info=table_names,
        top_k=10000,
        today_date=date.today())
    
    output = db_chain(final_prompt)

    chatbot_history = []

    chatbot_history.append(
        (
            f"{question}",
            f"{output['result']}"
        )
    )

    return output



def resummarise(query, initial_query, initial_summary, llm):

    """ Create a database query chain using the VertexAI LLM.

    Arguments:
        LLM: The large language model
        query: The question that is translated to an SQL query
        initial_query: The result query produced by the LLM
        initial_summary: The result summary produced by LLM


     Returns:
        query: The question that is translated to an SQL query
        initial_query: The result query produced by the LLM
        initial_summary: The result summary produced by LLM


    """
        
    query_chain = LLMChain(
        llm=llm, prompt=prompt_resummarise, output_key="output")

    return query_chain.run(
        {
            "query": query,
            "initial_query": initial_query,
            "initial_summary": initial_summary,
        }
    )


def create_agent(llm, db, question):

    """ Create an SQL agent using the VertexAI LLM and Langchain.

    Arguments:
        LLM: The large language model
        Database (object): The database containing our knowledge
        condense_question_prompt (PromptTemplate):The prompt template used to prompt engineer our LLM to respond in a certain tone, etc.
    Returns:
        output: Answer returned the LLM chain 
    """
    # Define our agent’s toolkit which will be used to answer the user question
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    qa_chain = create_sql_agent(
    llm=llm,
    db=db, 
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    early_stopping_method="generate",
    )

    answer = qa_chain.run(question)

    return answer


def pandas_agent(question,llm, df):

    """ Create a pandas agent using the VertexAI LLM and Langchain.

    Arguments:
        LLM: The large language model
        Dataframe (object): The pandas dataframe gotten from manipulating the data 
    Returns:
        response: Answer returned the LLM chain 
    """
    
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    response = agent.run(question)
    return response



def agent_with_memory(question, db, llm, dialect):

    """ Create an SQL agent with memory using the VertexAI LLM and Langchain.

    Arguments:
        LLM: The large language model
        Dataframe (object): The pandas dataframe gotten from manipulating the data 
    Returns:
        response: Answer returned the LLM chain 
    """

    chat_history = []

    SQL_SUFFIX = """Begin! You are a chatbot having a conversation with a human

    History: {chat_history}
    Question: {input}
    Thought: Query the schema of the most relevant tables and return the correct answer
    {agent_scratchpad}"""

    agent = create_sql_agent(
        llm,
        SQLDatabaseToolkit(db=db, llm=llm),
        AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={
            "memory": ConversationBufferMemory(
                input_key="input", memory_key="chat_history", return_messages=True
            )
        },
        suffix=SQL_SUFFIX,
        input_variables=["input", "chat_history", "agent_scratchpad"],
        verbose=True,
        handle_parsing_errors=True
    )

    response = agent.run({"dialect": dialect, "input": question, "chat_history": [], "agent_scratchpad": {}})

    return response