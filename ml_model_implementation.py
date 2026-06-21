import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



data = pd.read_csv("spam.csv")

print("Dataset Preview:")
print(data.head())

X = data["message"]
y = data["label"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Training Completed")


predictions = model.predict(X_test)



accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy Score:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


while True:

    user_input = input(
        "\nEnter Email Text (or type 'exit'): "
    )

    if user_input.lower() == "exit":
        break

    sample_vector = vectorizer.transform(
        [user_input]
    )

    result = model.predict(sample_vector)

    print("Prediction:", result[0])

print("\nProgram Finished")
