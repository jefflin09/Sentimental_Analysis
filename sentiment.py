import kagglehub
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Download and Load Dataset
# --------------------------------------------------
path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

df = pd.read_csv(path + "/IMDB Dataset.csv")

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

# ==================================================
# PART 1: MACHINE LEARNING MODEL
# ==================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

ml_model = LogisticRegression(max_iter=1000)
ml_model.fit(X_train_tfidf, y_train)

ml_accuracy = accuracy_score(
    y_test, ml_model.predict(X_test_tfidf)
)

print("ML Model Accuracy:", ml_accuracy)

# ==================================================
# PART 2: DEEP LEARNING MODEL
# ==================================================

max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

dl_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

dl_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

dl_model.fit(
    X_train_pad,
    y_train,
    epochs=2,
    batch_size=128,
    validation_split=0.2
)

dl_accuracy = dl_model.evaluate(X_test_pad, y_test)[1]
print("DL Model Accuracy:", dl_accuracy)

# ==================================================
# STATEMENT TYPE PREDICTION (USER INPUT)
# ==================================================

def predict_statement_type(statement):
    # ML Prediction
    tfidf_input = vectorizer.transform([statement])
    ml_pred = ml_model.predict(tfidf_input)[0]

    # DL Prediction
    seq_input = tokenizer.texts_to_sequences([statement])
    pad_input = pad_sequences(seq_input, maxlen=max_len)
    dl_pred = int((dl_model.predict(pad_input) > 0.5)[0][0])

    ml_result = "Positive Statement" if ml_pred == 1 else "Negative Statement"
    dl_result = "Positive Statement" if dl_pred == 1 else "Negative Statement"

    return ml_result, dl_result


# --------------------------------------------------
# Take Statement from User
# --------------------------------------------------
user_statement = input("\nEnter a statement: ")

ml_output, dl_output = predict_statement_type(user_statement)

print("\nStatement Analysis Result")
print("Machine Learning Model:", ml_output)
print("Deep Learning Model:", dl_output)