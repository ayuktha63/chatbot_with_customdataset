from flask import Flask, request, jsonify, render_template
from datetime import datetime
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# Function to load dataset from JSONL file
def load_dataset(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions.append(data['question'])
            answers.append(data['answer'])
    return questions, answers

# Load dataset
file_path = 'abhinav.jsonl'  # Replace with your actual file path
questions, answers = load_dataset(file_path)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
question_sequences = tokenizer.texts_to_sequences(questions)

# Pad sequences for consistent input size
max_len = max(len(seq) for seq in question_sequences)
padded_questions = pad_sequences(question_sequences, maxlen=max_len, padding='post')

# Model architecture
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(len(answers), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(padded_questions, np.arange(len(answers)), epochs=800, verbose=1)

# Load pre-trained semantic similarity model
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
corpus_embeddings = similarity_model.encode(questions, convert_to_tensor=True)

# Function to format the answer for HTML
def format_answer(answer):
    # Replace newlines with <br> and format as bullet points
    answer = answer.replace('\n', '<br>').strip()
    return '<ul><li>' + '</li><li>'.join(answer.split('<br>')) + '</li></ul>'

# Function to predict and return formatted answer
def get_answer(user_input, name="Krishna", email="krishna@gmail.com"):
    # Normalize user input for better matching
    normalized_input = user_input.lower().strip()

    # List of greetings and specific queries
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'thank you', 'welcome', 'thanks', 'bye', 'tata']
    time_queries = ['what is the time now', 'current time','time','what is the time','what is the current time']

    bot_name = 'zeebot'

    # Check if user input matches any greetings or specific queries
    if normalized_input in time_queries:
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}"

    for greeting in greetings:
        if normalized_input == greeting:
            if greeting == "hello":
                return f"Hello! How can I assist you today?"
            elif greeting == "hi":
                return f"Hi! How can I assist you today?"
            elif greeting == "hey":
                return f"Hey! How can I assist you today?"
            elif greeting == "good morning":
                return f"Good morning! How can I assist you today?"
            elif greeting == "thank you" or greeting == "thanks" or greeting == "welcome":
                return f"You're welcome! I will be here if you have any queries. Bye!"
            elif greeting == "bye" or "tata":
                return f"Goodbye! Have a great day!"
            elif greeting == "good afternoon":
                return f"Good afternoon! How can I assist you today?"
            elif greeting == "good evening":
                return f"Good evening! How can I assist you today?"

    # Check if user input includes any additional request with greetings
    for greeting in greetings:
        if greeting in normalized_input:
            if "your name" in normalized_input:
                return f"Hello! My name is {bot_name}. What can I do for you?"
            elif "my name" in normalized_input:
                return f"Hello! Your name is {name}. What can I do for you?"
            elif "my email" in normalized_input:
                return f"Hello! Your email is {email}. What can I do for you?"
            elif "help" in normalized_input or "help me" in normalized_input:
                return "Ok, let me help you. Please provide details about the issue."

    # Use the semantic similarity model to find the closest question
    query_embedding = similarity_model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)
    closest_n = torch.topk(similarities, k=1)
    closest_question_index = closest_n[1][0].item()
    
    # Get the answer and format it
    if similarities[0][closest_question_index] < 0.5:  # Threshold for out-of-scope questions
        if "your name" in normalized_input:
            return f"My name is {bot_name}."
        elif "my name" in normalized_input:
            return f"Your name is {name}."
        elif "my email" in normalized_input:
            return f"Your email is {email}."
        else:
            return "I am not getting the question correctly or it is not related to my context. Please rephrase the question."
    else:
        answer = answers[closest_question_index]
        formatted_answer = format_answer(answer)
        return formatted_answer


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def answer():
    user_input = request.json.get('question')
    predicted_answer = get_answer(user_input)
    return jsonify({'answer': predicted_answer})

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Change the port number to 8080 or any other desired port
