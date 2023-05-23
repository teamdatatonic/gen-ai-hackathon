import os
import requests
import gradio as gr
import google.auth.transport.requests
import google.oauth2.id_token

BACKEND_URL = os.environ["BACKEND_URL"]


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


def q_a(question: str, history: list):
    response = _query_model(question, history)

    # Format source documents (sources of excerpts passed to the LLM) into links the user can validate
    sources = [
        "[{0}]({0})".format(doc.metadata["source"])
        for doc in response["source_documents"]
    ]

    # Return the LLM answer, and list of sources used (formatted as a string)
    return response["answer"], "\n\n".join(sources)


def _query_model(question, history):
    qa_req_body = {
        "question": question,
        "chat_history": history,
    }
    return requests.post(
        url=BACKEND_URL + "/query_model",
        headers=_make_request_headers(),
        json=qa_req_body,
    ).json()


def _make_request_headers():
    try:
        request = google.auth.transport.requests.Request()
        audience = BACKEND_URL
        token = google.oauth2.id_token.fetch_id_token(request, audience)
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
    except Exception as error:
        print(error)
        return {
            "Content-Type": "application/json",
        }


def main():
    # Build a simple GradIO app that accepts user input and queries the LLM
    # Then displays the response in a ChatBot interface, with markdown support.
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        

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


if __name__ == "__main__":
    main()
