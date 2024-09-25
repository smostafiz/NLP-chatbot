from stewie_gpt2model import generate_gpt2_response, load_gpt2_model
from stewie_prepossessing_cleaning import max_encoder_seq_length, input_features_dict
from stewie_training_eval import num_encoder_tokens, decode_response
import numpy as np
from keras.models import load_model
import re
import datetime


class ChatBot:
    """Define exit commands."""
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def __init__(self):
        """This function initializes a ChatBot instance. It loads two essential models: the GPT-2 language model
        and tokenizer, as well as a Seq2Seq model. Additionally, it opens a chatlog file for logging interactions.
        This function sets up the ChatBot, making it ready to generate responses and log conversations with users."""
        self.t_model, self.tokenizer = load_gpt2_model()
        self.seq2seq_model = load_model('stewie_seq2seq_model.h5')
        self.chatlog_file = open("chatlog.txt", "a")

    def __del__(self):
        """This function is responsible for cleanup when a ChatBot instance is deleted."""
        self.chatlog_file.close()

    def log_chat(self, user_input, response):
        """The log_chat function records user interactions with the chatbot by logging the user's input
        and the bot's response in a chatlog file. It adds a timestamp to each entry, providing a chronological record
        of the conversation."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chatlog_file.write(f"{timestamp} - User: {user_input}\n")
        self.chatlog_file.write(f"{timestamp} - STEWIE: {response}\n")

    def chat(self):
        """The chat function initiates a conversation with the chatbot. It starts by displaying a welcome message
        and then enters a loop to facilitate user interactions. Within the loop, the chatbot prompts the user for input,
        checks if the input contains any exit commands to terminate the chat, generates a response
        using the Seq2Seq model, displays the response to the user, and logs both the user's input
        and the chatbot's response."""
        print("STEWIE: Hello! How can I assist you today?")

        while True:
            user_input = input("You: ")

            if any(cmd in user_input.lower() for cmd in self.exit_commands):
                print("STEWIE: Goodbye!")
                break

            response = self.generate_response(user_input)
            print(f"STEWIE: {response}")
            self.log_chat(user_input, response)

    def string_to_matrix(self, user_input):
        """The string_to_matrix function converts user input text into a binary matrix representation suitable
        for input to a machine learning model. It tokenizes the user input, initializes a matrix with zeros
        to represent input features, and then iterates through the tokens, setting corresponding features to 1 based on
        a predefined dictionary. The resulting binary input matrix is returned, providing a structured input format
        for further processing by the model."""
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')

        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    def make_exit(self, reply):
        """This function examines a given response to determine if it contains an exit command. It iterates
        through a predefined list of exit commands and checks if any of them appear in the reply. If an exit command
        is found, it prints a farewell message and returns True to signal that the chat session should end. If no
        exit command is detected, it returns False, indicating that the conversation should continue."""
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
        return False

    def generate_response(self, input_text):
        """This function processes user input and generates responses for the chatbot. It begins by preprocessing
        the input text and obtaining a response using a Seq2Seq model. It checks if the Seq2Seq response is empty
        or generic, or if it contains negative phrases. If any of these conditions are met, it switches to using
        a GPT-2 model to generate a more meaningful response. If the Seq2Seq response is deemed appropriate,
        it removes '<START>' and '<END>' tokens and returns the response."""
        input_matrix = self.string_to_matrix(input_text)
        seq2seq_response = decode_response(input_matrix)
        negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

        if not seq2seq_response.strip() or seq2seq_response.lower() in negative_responses:
            gpt2_response = generate_gpt2_response(input_text, self.t_model, self.tokenizer)
            return gpt2_response

        seq2seq_response = seq2seq_response.replace("<START>", '')
        seq2seq_response = seq2seq_response.replace("<END>", '')
        return seq2seq_response


"""Main program."""
if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.chat()
