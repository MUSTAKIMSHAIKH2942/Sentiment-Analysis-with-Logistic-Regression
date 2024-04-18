import pandas as pd
import re
import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load sample chat logs into a DataFrame
chat_logs = [
    {"customer_id": 1, "timestamp": "2024-04-16 10:05:00", "message": "Hello, I have an issue with my order."},
    {"customer_id": 2, "timestamp": "2024-04-16 10:10:00", "message": "Hi there, my package hasn't arrived yet."},
    {"customer_id": 3, "timestamp": "2024-04-16 10:15:00", "message": "Good morning, I need help with my account."},
    {"customer_id": 1, "timestamp": "2024-04-16 10:20:00", "message": "Sure, what seems to be the problem?"},
    {"customer_id": 2, "timestamp": "2024-04-16 10:25:00", "message": "It's been two weeks since I ordered."},
    {"customer_id": 3, "timestamp": "2024-04-16 10:30:00", "message": "I can't log in to my account."},
    {"customer_id": 1, "timestamp": "2024-04-16 10:35:00", "message": "Let me check your order status."},
    {"customer_id": 2, "timestamp": "2024-04-16 10:40:00", "message": "This is unacceptable. I need a refund."},
    {"customer_id": 3, "timestamp": "2024-04-16 10:45:00", "message": "I've tried resetting my password, but it's not working."},
    {"customer_id": 1, "timestamp": "2024-04-16 10:50:00", "message": "Your order is in transit. It should arrive soon."},
    {"customer_id": 2, "timestamp": "2024-04-16 10:55:00", "message": "I want to speak to a manager."},
    {"customer_id": 3, "timestamp": "2024-04-16 11:00:00", "message": "Thank you, I'll try again."},
]
chat_df = pd.DataFrame(chat_logs)

# Function to preprocess text data using spaCy
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and numbers using regex
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Lemmatize words using spaCy
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    # Join tokens back into a string
    preprocessed_text = " ".join(lemmatized_tokens)
    return preprocessed_text

# Apply preprocessing to the 'message' column
chat_df['clean_message'] = chat_df['message'].apply(preprocess_text)

# Display the preprocessed chat logs
print(chat_df[['customer_id', 'timestamp', 'message', 'clean_message']])
