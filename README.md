# SpamFilter Package

The **spamfilter** package provides a simple implementation of a Naive Bayes classifier for detecting spam messages. This package includes tools for training the classifier and perform predictions on unseen data. It also includes extra functionalities to cleanse and transform data.

## Features
- **NaiveBayes Classifier:** Implements the Naive Bayes algorithm with Laplace smoothing for spam detection.
- **Train and Test Integration:** Supports loading and processing of training and testing datasets from CSV files.
- **Manual Accuracy and Confusion Matrix Calculation:** Includes custom functions for computing evaluation metrics without relying on external libraries.
- **Customization:** Allows users to specify the smoothing parameter (alpha) during training.

## Author and Maintainer
- **Author:** Aaron Castillo
- **Maintainer:** Aaron Castillo
- **Contact:** your.email@example.com

## Version
**Current Version:** 0.0.1

## Installation
Clone the repository to your local machine:
```bash
```
Navigate to the project directory:
```bash
cd spamfilter
```

## Usage

### 1. Training the Classifier
To train the classifier, ensure you have a labeled dataset in CSV format (e.g., `train.csv`) with the following columns:
- `message`: The text of the message.
- `label`: The label indicating whether the message is "spam" or "ham".

Example training script:
```python
from classifier import NaiveBayes
from filter import load_data

# Load training data
word_counts, labels = load_data("train.csv")

# Initialize and train the classifier
classifier = NaiveBayes(word_counts, labels)
classifier.fit(alpha=1)

# Print the classifier's parameters
print(classifier)
```

### 2. Testing the Classifier
To test the classifier, provide another labeled dataset in CSV format (e.g., `test.csv`). Example testing script:
```python
from filter import load_data, calculate_confusion_matrix_and_accuracy

# Load test data
word_counts, labels = load_data("test.csv")

# Classify messages
predictions = [classifier.classify(msg) for msg in pd.read_csv("test.csv")['message']]

# Evaluate the classifier
true_labels = ["spam" if label == 1 else "ham" for label in labels]
confusion_matrix, accuracy = calculate_confusion_matrix_and_accuracy(true_labels, predictions)

print("Confusion Matrix:")
for true_label, row in confusion_matrix.items():
    print(f"{true_label}: {row}")
print(f"Accuracy: {accuracy:.2f}")
```

### 3. Example Output
**Confusion Matrix (Train Data):**
```
ham: {'ham': 50, 'spam': 2}
spam: {'ham': 3, 'spam': 45}
```
**Accuracy:**
```
0.95
```

## File Structure
```
spamfilter/
├── classifier.py  # Contains the NaiveBayes class
├── filter.py      # Script for training, testing, and evaluation
├── train.csv      # Sample training dataset
├── test.csv       # Sample testing dataset
└── README.md      # Documentation
```

## Requirements
- Python 3.7+
- pandas
- numpy 

Install dependencies with:
```bash
pip install pandas
```

## For Package Development
Need to install:
- `hatchling`
- `build`
- `wheel`
- `pytest`
- (optional) `setuptools`

## License
This project is licensed under the MIT License. See the LICENSE file for details.