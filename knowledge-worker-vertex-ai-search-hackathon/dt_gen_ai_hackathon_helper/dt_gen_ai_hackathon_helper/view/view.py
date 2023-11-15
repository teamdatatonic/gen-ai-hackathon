import gradio as gr
from langchain.prompts import PromptTemplate

from dt_gen_ai_hackathon_helper.chains.chains import create_qa_chain
from dt_gen_ai_hackathon_helper.formatter_helper.formatter_helper import gsutil_uri_to_gcs_url
from dt_gen_ai_hackathon_helper.prompts.prompts import TASK_01_PROMPT


class View:
    def __init__(self, qa_chain=None, vector_store=None):
        if vector_store:
            condense_question_prompt = PromptTemplate.from_template(TASK_01_PROMPT)
            self.qa_chain = create_qa_chain(vector_store, condense_question_prompt)
        elif qa_chain:
            self.qa_chain = qa_chain
        else:
            raise Exception("Must pass either qa_chain or vector_store as argument.")

    def q_a(self, question: str, history: list):
        # map history (list of lists) to expected format of chat_history (list of tuples)
        chat_history = map(tuple, history)

        # Query the LLM to get a response
        # First the Q&A chain will collect documents semantically similar to the question
        # Then it will ask the LLM to use this data to answer the user question
        # We also provide chat history as further context
        response = self.qa_chain(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        # Format source documents (sources of excerpts passed to the LLM) into links the user can validate
        # Strip index.html so URLs terminate in the parent folder
        # Strip https:// and http:// and replace with https:// to enforce https protocol and catch cases where
        # https:// is present or not present
        source_gcs_urls = [
            gsutil_uri_to_gcs_url(doc.metadata['source']) if doc.metadata['source'].startswith("gs://")
            else f"https://{doc.metadata['source'].replace('index.html', '').replace('https://', '').replace('http://', '')}"
            for doc in response["source_documents"]
        ]
        # Return the LLM answer, and list of sources used (formatted as a string)
        return response["answer"], "\n\n".join(source_gcs_urls)

    def submit(self, msg, chatbot):
        # First create a new entry in the conversation log
        msg, chatbot = self.user(msg, chatbot)
        # Then get the chatbot response to the user question
        chatbot = self.bot(chatbot)
        return msg, chatbot

    def user(self, user_message, history):
        # Return "" to clear the user input, and add the user question to the conversation history
        return "", history + [[user_message, None]]

    def bot(self, history):
        # Get the user question from conversation history
        user_message = history[-1][0]
        # Get the response and sources used to answer the user question
        bot_message, bot_sources = self.q_a(user_message, history[:-1])

        # Using a template, format the response and sources together
        bot_template = (
            "{0}\n\n<details><summary><b>Sources</b></summary>\n\n{1}</details>"
        )
        # Place the response into the conversation history and return
        history[-1][1] = bot_template.format(bot_message, bot_sources)
        return history

    def launch_interface(self, share=True, debug=True):
        # Build a simple GradIO app that accepts user input and queries the LLM
        # Then displays the response in a ChatBot interface, with markdown support.
        with gr.Blocks(theme=gr.themes.Base()) as demo:
            # Set a page title
            gr.Markdown("# Custom knowledge worker")
            # Create a chatbot conversation log
            chatbot = gr.Chatbot(label="🤖 knowledge worker")
            # Create a textbox for user questions
            msg = gr.Textbox(
                label="👩‍💻 user input",
                info="Query information from the custom knowledge base.",
            )

            # Align both buttons on the same row
            with gr.Row():
                send = gr.Button(value="Send", variant="primary", size="sm")
                clear = gr.Button(value="Clear History", variant="secondary", size="sm")

            # Submit message on <enter> or clicking "Send" button
            msg.submit(self.submit, [msg, chatbot], [msg, chatbot], queue=False)
            send.click(self.submit, [msg, chatbot], [msg, chatbot], queue=False)

            # Clear chatbot history on clicking "Clear History" button
            clear.click(lambda: None, None, chatbot, queue=False)

        # Create a queue system so multiple users can access the page at once
        demo.queue()
        # Launch the webserver locally
        demo.launch(share=share, debug=debug)
