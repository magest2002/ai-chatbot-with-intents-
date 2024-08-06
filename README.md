# Chatbot Model Training

This repository contains code for training a simple neural network-based chatbot model using Keras and natural language processing (NLP) techniques.

## Overview

The project involves preprocessing text data from an `intents.json` file, creating a neural network model to classify user intents, training the model on the processed data, and saving the trained model for use in a chatbot application.

## Requirements

- Python 3.x
- Keras
- TensorFlow
- NumPy
- NLTK
- Pickle
- JSON

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chatbot-model-training.git
    cd chatbot-model-training
    ```

2. Install the required libraries:
    ```bash
    pip install tensorflow keras numpy nltk
    ```

3. Download NLTK data:
    ```python
    import nltk
    nltk.download('omw-1.4')
    nltk.download("punkt")
    nltk.download("wordnet")
    ```

## Dataset

Ensure you have an `intents.json` file containing your chatbot's intents, patterns, and responses. Place this file in the root directory of the project.

## Training the Model

1. Run the training script:
    ```bash
    python train_chatbot.py
    ```

2. The script will preprocess the data, create and train the model, and save the trained model as `chatbot_model.h5`.

## Model Architecture

The model consists of three layers:
- Input layer with 128 neurons and ReLU activation
- Hidden layer with 64 neurons and ReLU activation
- Output layer with softmax activation and a number of neurons equal to the number of intents

## Usage

Once the model is trained, you can use it in a chatbot application to classify user intents and generate appropriate responses.
