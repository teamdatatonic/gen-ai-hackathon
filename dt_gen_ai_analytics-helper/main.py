from dt_gen_ai_analytics_helper.chains.chains import create_db_chain
import dt_gen_ai_analytics_helper.view.view as demo_view
import gradio as gr


# Define a function to query the SQLDBChain
def query_database(question):
    # Call the SQLDBChain to get the answer based on the question
    qa_chain = create_db_chain()
    answer = qa_chain(question)


    return answer

# Create a Gradio interface
iface = gr.Interface(
    fn=query_database,  # Function to execute when a query is received
    inputs="text",      # Input is a single text field
    outputs="json",     # Output will be a JSON response
    title="Analytics Worker Demo",
    description="Enter a question, and the system will query the database and provide an answer.",
)

# Launch the Gradio interface on a specified port (e.g., 5000)
iface.launch(share=True)


# run app
if __name__ == "__main__":
    query_database()