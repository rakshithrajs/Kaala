"""Classifier for detecting goals or intents in text."""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

class ImprovedGoalClassifier:
    def __init__(self, max_features=3000):
        """
        Improved Fast Logistic Regression-based Goal/Intent Classifier
        """
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 3), 
            lowercase=True,
            strip_accents='ascii',
            min_df=1,  
            max_df=0.95
        )

        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=2000,
            solver='liblinear'
        )

        self.is_trained = False
        self.optimal_threshold = 0.5 

    def extract_features(self, text:str) -> dict:
        """ Extract features from text to identify goal indicators.

        Args:
            text (str): Text to analyze for goal indicators.

        Returns:
            dict: Dictionary containing counts of goal-related words and patterns.
        """
        text = text.lower()

        goal_words = ['want', 'need', 'should', 'have to', 'must', 'plan', 'goal', 'aim', 'try', 'wish', 'hope']
        action_words = ['do', 'make', 'create', 'build', 'learn', 'study', 'work', 'practice', 'improve', 'get']
        request_words = ['help', 'assist', 'remind', 'schedule', 'set', 'call', 'contact', 'book', 'arrange']
        future_words = ['will', 'going to', 'tomorrow', 'next', 'soon', 'later', 'tonight', 'today']

        goal_count = sum(1 for word in goal_words if word in text)
        action_count = sum(1 for word in action_words if word in text)
        request_count = sum(1 for word in request_words if word in text)
        future_count = sum(1 for word in future_words if word in text)

        has_modal = bool(re.search(r'\b(can|could|would|should|might|may)\b', text))
        has_question = '?' in text
        has_imperative = text.startswith(('remind', 'help', 'schedule', 'set', 'call', 'please'))
        has_personal_goal = bool(re.search(r'\b(i|my|me)\s+(want|need|should|have to|must|plan|goal|aim)', text))

        return {
            'goal_words': goal_count,
            'action_words': action_count, 
            'request_words': request_count,
            'future_words': future_count,
            'has_modal': has_modal,
            'has_question': has_question,
            'has_imperative': has_imperative,
            'has_personal_goal': has_personal_goal,
            'length': len(text.split())
        }

    def preprocess_text(self, text:str) -> str:
        """ Preprocess text for classification by normalizing and adding feature indicators.

        Args:
            text (str): Text to preprocess.

        Returns:
            str: Preprocessed text with feature indicators.
        """
        text = re.sub(r'\s+', ' ', text.lower().strip())

        features = self.extract_features(text)

        feature_tokens = []
        if features['has_personal_goal']:
            feature_tokens.append('PERSONAL_GOAL_INDICATOR')
        if features['has_imperative']:
            feature_tokens.append('IMPERATIVE_INDICATOR')
        if features['goal_words'] > 0:
            feature_tokens.append('GOAL_WORDS_PRESENT')
        if features['action_words'] > 1:
            feature_tokens.append('MULTIPLE_ACTIONS')

        if feature_tokens:
            text += ' ' + ' '.join(feature_tokens)

        return text

    def load_dataset(self, csv_path='goal_intent_dataset.csv') -> tuple:
        df = pd.read_csv(csv_path)
        texts = [self.preprocess_text(text) for text in df['sentence']]
        labels = df['goal_detected'].values
        return texts, labels

    def find_optimal_threshold(self, X_val, y_val):
        """ Find the optimal classification threshold based on F1 score.

        Args:
            X_val (int): Validation feature matrix.
            y_val (int): Validation labels.

        Returns:
            int: Optimal threshold for classification.
        """
        probabilities = self.classifier.predict_proba(X_val)[:, 1]

        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.3, 0.8, 0.05):
            predictions = (probabilities >= threshold).astype(int)

            tp = np.sum((predictions == 1) & (y_val == 1))
            fp = np.sum((predictions == 1) & (y_val == 0))
            fn = np.sum((predictions == 0) & (y_val == 1))

            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0

            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def train(self, csv_path='goal_intent_dataset.csv', test_size=0.2, val_size=0.1) -> float:
        """ Train the classifier on the dataset.

        Args:
            csv_path (str, optional): Dataset Path. Defaults to 'goal_intent_dataset.csv'.
            test_size (float, optional): The percentage of exmaples you want as test set. Defaults to 0.2.
            val_size (float, optional): The percetage of exmaples you want as validation set. Defaults to 0.1.

        Returns:
            float: Test accuracy of the trained classifier.
        """
        texts, labels = self.load_dataset(csv_path)
        
        X = self.vectorizer.fit_transform(texts)

        labels_np = np.array(labels)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, labels_np, test_size=test_size, random_state=42, stratify=labels_np
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )

        self.classifier.fit(X_train, y_train)

        self.optimal_threshold = self.find_optimal_threshold(X_val, y_val)
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")

        test_probabilities = self.classifier.predict_proba(X_test)[:, 1]
        y_pred = (test_probabilities >= self.optimal_threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Training examples: {X_train.shape[0]}")
        print(f"Validation examples: {X_val.shape[0]}")
        print(f"Test examples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}")

        print("\nClassification Report (with optimized threshold):")
        print(classification_report(y_test, y_pred, target_names=['No Goal', 'Has Goal']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

        self.is_trained = True
        return float(accuracy)

    def predict(self, text:str, use_optimal_threshold=True) -> dict:
        """ Predict if the input text contains a goal or intent.

        Args:
            text (str): Text to classify.
            use_optimal_threshold (bool, optional): Defaults to True.

        Raises:
            ValueError: If the classifier has not been trained yet.

        Returns:
            dict: Prediction results including text, goal presence, confidence, and probabilities.
        """
        if not self.is_trained:
            self.train()
            self.is_trained = True

        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        probabilities = self.classifier.predict_proba(X)[0]
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        prediction = (probabilities[1] >= threshold).astype(int)

        return {
            "text": text,
            "has_goal": 1 if bool(prediction) else 0,
            "confidence": float(max(probabilities)),
            "goal_probability": float(probabilities[1]),
            "threshold_used": threshold,
            "probabilities": {
                "no_goal": float(probabilities[0]),
                "has_goal": float(probabilities[1])
            }
        }

    def save_model(self, filepath='improved_goal_classifier.pkl'):
        """ Save the trained model to a file.

        Args:
            filepath (str, optional): File path. Defaults to 'improved_goal_classifier.pkl'.

        Raises:
            ValueError: If the model has not been trained yet or if the file path is invalid.
        """
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'optimal_threshold': self.optimal_threshold,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Improved model saved to {filepath}")

    def load_model(self, filepath='improved_goal_classifier.pkl'):
        """ Load a trained model from a file.

        Args:
            filepath (str, optional): File path. Defaults to 'improved_goal_classifier.pkl'.

        Raises:
            ValueError: If the file path is invalid or if the file does not contain a valid model.
        """
        if not filepath.endswith('.pkl'):
            raise ValueError("Model file must be a .pkl file")
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.optimal_threshold = model_data['optimal_threshold']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

# Usage example
if __name__ == "__main__":
    # Initialize and train the improved classifier
    classifier = ImprovedGoalClassifier()

    # Train with your dataset
    accuracy = classifier.train('goal_intent_dataset.csv')

    # Save the trained model
    classifier.save_model('improved_goal_classifier.pkl')

    # Test with examples
    test_messages = [
        "I need to finish my homework tonight",
        "How are you doing today?", 
        "Can you remind me to call my mom?",
        "The weather is nice",
        "I want to learn guitar",
        "gotta study for the exam",
        "hey what's up",
        "I should workout more",
        "nice weather today",
        "help me plan my schedule"
    ]

    print("\n" + "="*70)
    print("TESTING THE IMPROVED CLASSIFIER")
    print("="*70)

    for message in test_messages:
        result = classifier.predict(message)
        print(f"Message: '{message}'")
        print(f"Has Goal: {result['has_goal']}")
        print(f"Goal Probability: {result['goal_probability']:.3f}")
        print(f"Threshold: {result['threshold_used']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 50)

    print("\n" + "="*70)
    print("ROUTING EXAMPLES WITH IMPROVED CLASSIFIER")
    print("="*70)
