# from langchain.chains import SQLDatabaseSequentialChain
from langchain_experimental.sql.base import  SQLDatabaseSequentialChain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import(create_pandas_dataframe_agent)
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent 
from langchain import LLMChain,PromptTemplate
from datetime import date
from pathlib import Path
import os



def load_prompt_template(path, input_variables):
    with open(str(path), "r") as fp:
        return PromptTemplate(template=fp.read(), input_variables=input_variables)


prompts = Path(os.path.realpath(__file__)).parent.parent / "prompts"
prompt_generate_sql = load_prompt_template(
    prompts / "google_sql.txt", ["question", "table_info", "top_k", "today_date"]
)
prompt_resummarise = load_prompt_template(
    prompts / "resummarise.txt",
    ["query", "initial_summary", "initial_query", "initial_data"],
)
table_names = "customers,orders,employees,inventory,product_reviews,supplier_orders,financial_goals,marketing_campaign_analytics"



def create_db_chain(llm, db, question):
    """ Create a Q&A conversation chain using the VertexAI LLM.

    """
    
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm,
        db,
        verbose=True,
        return_intermediate_steps=True,
    )
    # test_prompt = PromptTemplate(template=TEST_PROMPT, input_variables=["question"])
 
    final_prompt = prompt_generate_sql.format(
        question=question,
        table_info=table_names,
        top_k=10000,
        today_date=date.today())
    output = db_chain(final_prompt)
    # return output['result']
    return output



def resummarise(query, initial_query, initial_summary, initial_data, llm):
        
    query_chain = LLMChain(
        llm=llm, prompt=prompt_resummarise, output_key="output")

    return query_chain.run(
        {
            "query": query,
            "initial_query": initial_query,
            "initial_data": initial_data,
            "initial_summary": initial_summary,
        }
    )


def create_agent(llm, db, question):
    # Define our agent’s toolkit which will be used to answer the user question
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    qa_chain = create_sql_agent(
    llm=llm,
    db=db, 
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    early_stopping_method="generate")

    answer = qa_chain.run(question)

    return answer


def pandas_agent(question,llm, df):
    
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    response = agent.run(question)
    return response