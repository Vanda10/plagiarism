import string
from collections import Counter
import math
from sklearn.metrics import accuracy_score

## Remove punctuation and convert to lowercase
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

## Generate a list of ngram from text
def get_ngrams(text, n):
    tokens = text.split()
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return ngrams

## Calculate cosine similarity
def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1[key] * vector2.get(key, 0) for key in vector1)
    magnitude1 = math.sqrt(sum(vector1[key] ** 2 for key in vector1))
    magnitude2 = math.sqrt(sum(vector2[key] ** 2 for key in vector2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

## Use get_ngrams, count the ngram
## and use cosine_similarity to calculate the similarity
def calculate_similarity(doc1, doc2, n):
    doc1_ngrams = Counter(get_ngrams(doc1, n))
    doc2_ngrams = Counter(get_ngrams(doc2, n))
    return cosine_similarity(doc1_ngrams, doc2_ngrams)

##
def train_test_model(original_text, suspicious_text, n):
    original_documents = {'original.txt': preprocess_text(original_text)}
    suspicious_documents = {'suspicious.txt': preprocess_text(suspicious_text)}

    # Prepare data
    X = []
    y = []
    for suspicious_filename, suspicious_content in suspicious_documents.items():
        for original_filename, original_content in original_documents.items():
            similarity = calculate_similarity(suspicious_content, original_content, n)
            X.append((suspicious_content, original_content))
            # Label 1 if similarity is above threshold, 0 otherwise
            y.append(1 if similarity > 0.8 else 0)

    # Dummy classifier that predicts no plagiarism (0) for all cases
    y_pred_test = [0] * len(y)

    # Calculate accuracy
    test_accuracy = accuracy_score(y, y_pred_test)

    return test_accuracy, similarity