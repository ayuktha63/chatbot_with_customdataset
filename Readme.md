
# Chat Bot with Custom Data Set

## Project Overview
This project is a chat bot application built using a custom data set provided by a firm for testing purposes. The application uses the TensorFlow library in Python and is served by the Flask framework. The chat bot allows users to type questions, sends the questions to a trained model, and displays the responses in a chat box.

## Project Files
- **abhinav.jsonl**: The custom data set used to train the model.
- **app.py**: The main Python file containing the model and backend logic served by Flask.
- **static/Zeetius - Sports Management & Automation_files**: Contains CSS, JavaScript, and images for the website.
- **templates/index.html**: The main HTML file containing the front-end of the website.

## Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ayuktha63/chatbot_with_customdataset.git
    cd chatbot_with_customdataset
    ```

2. **Install the necessary libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    python app.py
    ```

## Features
- **Chat Bot Interface**: Users can type questions in the chat box and get responses from the trained model.
- **Custom Data Set**: The model is trained using a custom data set (`abhinav.jsonl`).
- **Flask Backend**: The backend is built using Flask, which handles the requests and responses.
- **TensorFlow Model**: The chat bot uses a TensorFlow model for generating responses.

## Project Outcome
The primary goal of this project was to build a functional chat bot and become familiar with the process of working with Python code, utilizing various libraries, and training a machine learning model. This project provided hands-on experience with:
- Setting up a Flask application.
- Using TensorFlow to train and utilize a model.
- Integrating front-end and back-end components.
- Working with custom data sets.

## Acknowledgements
This project was built with significant help from ChatGPT. The entire application, from front-end design to backend logic and model training, was developed using guidance and code snippets provided by ChatGPT.


Feel free to reach out if you have any questions or need further assistance with the setup or usage of this project.

---

# requirements.txt
```
Flask
numpy
jsonlib-python3
tensorflow
sentence-transformers
torch
```

This file lists all the required libraries and frameworks needed for your project. To install these dependencies, use the following command:

```bash
pip install -r requirements.txt
```
