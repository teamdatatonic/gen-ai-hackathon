## Datatonic GenAI Hackathon Helper module
- A helper Python module to abstract parts of the Datatonic Gen AI Hackathon workshop. 

## To run the gradio interface and query the LLM
- `poetry install`
- `cd dt_gen_ai_analytics-helper` then `make run`

## Some test questions for the LLMchain 
- How many employees are there?
- How many employees are there? write an email about this 
- What are the top 5 cities where my customers are located in
- What is the total_revenue?
 - Summarize this
 - why do they have the highest average rating?
- How many customers are there? and compose a summary about this 

## Some test questions for the SQLAgent
- Check for duplicate entries in the "product_reviews" table
- Identify any missing data in the "employees" table, such as missing contact information
- Determine the most popular product in terms of reviews over different time periods
- Evaluate the progress toward financial goals using the "financial_goals" table and compare it to actual financial data from other tables.
- Determine which suppliers have the most orders or provide the most products

## Some test questions for the PandasAgent
- List 10 rows in customer table?
- Create a bar chart to show top 2 product review scores
- List 3 customers and thier phone numbers