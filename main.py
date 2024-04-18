from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , accuracy_score ,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd 
# Step 1: Synthetic Dataset Generation (assuming 3 classes: positive, negative, neutral)
synthetic_chat_logs = [
    {"customer_id": 1, "timestamp": "2024-04-16 10:05:00", "message": "Hello, I have an issue with my order.", "sentiment": "negative"},
    {"customer_id": 2, "timestamp": "2024-04-16 10:10:00", "message": "Hi there, my package hasn't arrived yet.", "sentiment": "negative"},
    {"customer_id": 3, "timestamp": "2024-04-16 10:15:00", "message": "Good morning, I need help with my account.", "sentiment": "negative"},
    {"customer_id": 1, "timestamp": "2024-04-16 10:20:00", "message": "Sure, what seems to be the problem?", "sentiment": "neutral"},
    {"customer_id": 2, "timestamp": "2024-04-16 10:25:00", "message": "It's been two weeks since I ordered.", "sentiment": "negative"},
    {"customer_id": 3, "timestamp": "2024-04-16 10:30:00", "message": "I can't log in to my account.", "sentiment": "negative"},
    {"customer_id": 1, "timestamp": "2024-04-16 10:35:00", "message": "Let me check your order status.", "sentiment": "neutral"},
    {"customer_id": 2, "timestamp": "2024-04-16 10:40:00", "message": "This is unacceptable. I need a refund.", "sentiment": "negative"},
    {"customer_id": 3, "timestamp": "2024-04-16 10:45:00", "message": "I've tried resetting my password, but it's not working.", "sentiment": "negative"},
    {"customer_id": 1, "timestamp": "2024-04-16 10:50:00", "message": "Your order is in transit. It should arrive soon.", "sentiment": "positive"},
    {"customer_id": 2, "timestamp": "2024-04-16 10:55:00", "message": "I want to speak to a manager.", "sentiment": "negative"},
    {"customer_id": 3, "timestamp": "2024-04-16 11:00:00", "message": "Thank you, I'll try again.", "sentiment": "neutral"},
]


# Convert synthetic data to DataFrame
synthetic_df = pd.DataFrame(synthetic_chat_logs, columns=["message", "sentiment"])

# Step 2: Feature Engineering
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(synthetic_df["message"])
y = synthetic_df["sentiment"]

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = logistic_regression_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

