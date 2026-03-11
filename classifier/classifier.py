"""Classifier for detecting goals or intents in text."""

from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

_DATASET_PATH = Path(__file__).resolve().parent / "goal_intent_dataset.csv"
_MODEL_PATH = Path(__file__).resolve().parent / "best.pkl"


class ImprovedGoalClassifier:
    """Improved Goal/Intent classifier using Logistic Regression + TF-IDF."""

    def __init__(self, max_features=3000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 3),
            lowercase=True,
            strip_accents="ascii",
            min_df=1,
            max_df=0.95,
        )
        self.classifier = LogisticRegression(
            random_state=42, max_iter=2000, solver="liblinear"
        )
        self.is_trained = False
        self.optimal_threshold = 0.5

    def extract_features(self, text: str) -> dict:
        text = text.lower()
        goal_words = ["want", "need", "should", "have to", "must", "plan", "goal", "aim", "try", "wish", "hope"]
        action_words = ["do", "make", "create", "build", "learn", "study", "work", "practice", "improve", "get"]
        request_words = ["help", "assist", "remind", "schedule", "set", "call", "contact", "book", "arrange"]
        future_words = ["will", "going to", "tomorrow", "next", "soon", "later", "tonight", "today"]

        return {
            "goal_words": sum(1 for word in goal_words if word in text),
            "action_words": sum(1 for word in action_words if word in text),
            "request_words": sum(1 for word in request_words if word in text),
            "future_words": sum(1 for word in future_words if word in text),
            "has_modal": bool(re.search(r"\b(can|could|would|should|might|may)\b", text)),
            "has_question": "?" in text,
            "has_imperative": text.startswith(("remind", "help", "schedule", "set", "call", "please")),
            "has_personal_goal": bool(re.search(r"\b(i|my|me)\s+(want|need|should|have to|must|plan|goal|aim)", text)),
            "length": len(text.split()),
        }

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.lower().strip())
        features = self.extract_features(text)

        feature_tokens = []
        if features["has_personal_goal"]:
            feature_tokens.append("PERSONAL_GOAL_INDICATOR")
        if features["has_imperative"]:
            feature_tokens.append("IMPERATIVE_INDICATOR")
        if features["goal_words"] > 0:
            feature_tokens.append("GOAL_WORDS_PRESENT")
        if features["action_words"] > 1:
            feature_tokens.append("MULTIPLE_ACTIONS")

        if feature_tokens:
            text += " " + " ".join(feature_tokens)
        return text

    def load_dataset(self) -> tuple[list[str], np.ndarray]:
        df = pd.read_csv(_DATASET_PATH)
        texts = [self.preprocess_text(text) for text in df["sentence"]]
        labels = df["goal_detected"].values
        return texts, labels

    def find_optimal_threshold(self, x_val, y_val):
        probabilities = self.classifier.predict_proba(x_val)[:, 1]
        best_threshold, best_f1 = 0.5, 0

        for threshold in np.arange(0.3, 0.8, 0.05):
            predictions = (probabilities >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = (2 * (precision * recall) / (precision + recall)) if precision + recall > 0 else 0

            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold

        return best_threshold

    def train(self, test_size=0.2, val_size=0.1) -> float:
        texts, labels = self.load_dataset()
        x = self.vectorizer.fit_transform(texts)
        labels_np = np.array(labels)
        split_data = self._split_data(x, labels_np, test_size, val_size)

        self.classifier.fit(split_data["X_train"], split_data["y_train"])
        self.optimal_threshold = self.find_optimal_threshold(split_data["X_val"], split_data["y_val"])

        accuracy, y_pred = self._evaluate_model(split_data["X_test"], split_data["y_test"])
        self._print_reports(split_data, y_pred, accuracy)
        self.is_trained = True
        return float(accuracy)

    def _split_data(self, x, y, test_size, val_size):
        x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=y_temp,
        )
        return {"X_train": x_train, "X_val": x_val, "X_test": x_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}

    def _evaluate_model(self, x_test, y_test):
        test_probabilities = self.classifier.predict_proba(x_test)[:, 1]
        y_pred = (test_probabilities >= self.optimal_threshold).astype(int)
        return accuracy_score(y_test, y_pred), y_pred

    def _print_reports(self, data, y_pred, accuracy):
        y_test = data["y_test"]
        print("\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Training examples: {data['X_train'].shape[0]}")
        print(f"Validation examples: {data['X_val'].shape[0]}")
        print(f"Test examples: {data['X_test'].shape[0]}")
        print(f"Features: {data['X_train'].shape[1]}")
        print("\nClassification Report (with optimized threshold):")
        print(classification_report(y_test, y_pred, target_names=["No Goal", "Has Goal"]))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

    def predict(self, text: str, use_optimal_threshold=True) -> dict:
        if not self.is_trained:
            if self.load_model():
                print("Model loaded successfully.")
            else:
                self.train()
                self.is_trained = True

        processed_text = self.preprocess_text(text)
        x = self.vectorizer.transform([processed_text])
        probabilities = self.classifier.predict_proba(x)[0]
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        prediction = (probabilities[1] >= threshold).astype(int)

        return {
            "text": text,
            "has_goal": 1 if bool(prediction) else 0,
            "confidence": float(max(probabilities)),
            "goal_probability": float(probabilities[1]),
            "threshold_used": threshold,
            "probabilities": {"no_goal": float(probabilities[0]), "has_goal": float(probabilities[1])},
        }

    def save_model(self, filepath: str | os.PathLike = _MODEL_PATH):
        if not self.is_trained:
            raise ValueError("No trained model to save!")

        file = Path(filepath)
        if file.parent and not file.parent.exists():
            file.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "optimal_threshold": self.optimal_threshold,
            "is_trained": self.is_trained,
        }
        joblib.dump(model_data, file)
        print(f"Improved model saved to {file}")

    def load_model(self, filepath: str | os.PathLike = _MODEL_PATH):
        file = Path(filepath)
        if file.suffix != ".pkl":
            raise ValueError("Model file must be a .pkl file")
        if not file.exists():
            return False

        model_data = joblib.load(file)
        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.optimal_threshold = model_data["optimal_threshold"]
        self.is_trained = model_data["is_trained"]
        print(f"Model loaded from {file}")
        return self.is_trained
