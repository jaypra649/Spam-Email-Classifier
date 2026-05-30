# Import libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "message": [
        "Congratulations you won a prize",
        "Call me when you arrive",
        "Free money available now",
        "Let's meet tomorrow",
        "Win cash instantly",
        "How are you today",
        "Claim your free reward",
        "See you at college",
        "Limited offer click now",
        "Happy birthday friend"
    ],

    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

# Create dataframe
df = pd.DataFrame(data)

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Features and target
X = df["message"]
y = df["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Test a new message
new_message = ["Congratulations! You won free cash"]

new_vector = vectorizer.transform(new_message)

prediction = model.predict(new_vector)

if prediction[0] == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Ham")