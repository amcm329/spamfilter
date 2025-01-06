# SpamFilter Package

The **spamfilter** package provides a simple implementation of a Naive Bayes classifier for detecting spam messages. This package includes tools for training the classifier and perform predictions on unseen data. It also includes extra functionalities to cleanse and transform data.

## Features
- **NaiveBayes Classifier:** Implements the Naive Bayes algorithm with alpha smoothing for spam detection.
- **Train and Test Integration:** Supports loading and processing of training and testing datasets from CSV files.
- **Accuracy and Confusion Matrix Calculation:** Includes custom functions for computing these evaluation metrics.

## Author and Maintainer
- **Author:** Aaron Castillo
- **Maintainer:** Aaron Castillo
- **Contact:** amc224@imperial.ac.uk

## Version
**Current Version:** 0.0.1

## Installation
In the folder dist, theres a .tar.gz file, then you just need to execute: 
```bash
python pip install -m dist/spamfilter-0.0.1.tar.gz
```

## Usage
See the file filter.py for more details. 

## File Structure
```

spamfilter/
├── pyproject.toml #To build the package.
├── README.md #Documentation.
├── LICENSE #idem.
├── src/
│   └── spamfilter/
│       ├── __init__.py
│       ├── classifier.py #NaiveBayes classifier.
│       ├── utils.py #Auxiliary functions.
├── tests/
│   ├── test_classifier.py #Tests for NaiveBayes classifier.
│   ├── test_utils.py #Tests for auxiliary functions.
```

## Requirements
- Python 3.7+
- pandas 1.0.3+,
- numpy 1.18.4+

## For Package Development
Need to install:
- `hatchling`
- `build`
- `wheel`
- `pytest`
- `setuptools`

## License
This project is licensed under the MIT License. See the LICENSE file for details.
