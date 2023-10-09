import gradio as gr


class View:
    def __init__(self, qa_chain=None):
        if qa_chain:
            self.qa_chain = qa_chain
        else:
            raise Exception("Must pass either qa_chain.")
    
    def generate_query(self, question):
    
        if question:
            query = question
        else:
            raise Exception("Must pass either qa_chain.")

        # get SQL query from natural language
        output = self.qa_chain(query)

        return output
       

    def submit(self, msg, chatbot):
        # First create a new entry in the conversation log
        msg, chatbot = self.user(msg, chatbot)
        # Then get the chatbot response to the user question
        chatbot = self.bot(chatbot)
        return msg, chatbot

    def user(self, user_message):
        # Return "" to clear the user input, and add the user question to the conversation history
        return ""+ [[user_message, None]]
