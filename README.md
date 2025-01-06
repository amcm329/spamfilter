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

## Package Creation
You need to place the prompt in the main path of this folder, then execute: 
```bash
python -m pytest
python -m build
```

This will create a folder called *dist* with two files: 
* spamfilter-0.0.1.tar
* spamfilter-0.0.1-py3-none-any.whl 

## Installation
In the folder dist, theres a .tar.gz file, then you just need to execute: 
```bash
python pip install -m dist/spamfilter-0.0.1.tar.gz
```

Or:

```bash
python pip install -m dist/spamfilter-0.0.1-py3-none-any.whl
```

It is enough to execute only one option.

**_NOTE:_** Do not try to install the package by using a Python script or Python libraries, otherwise
it will create a partial installation, the recommended way is 

## Usage
See the file filter.py for more details, as it contains more nurtured examples.

## File Structure
```
spamfilter/
├── filter.py #Contains all implementations and analysis.
├── pyproject.toml #To build the package.
├── README.md #Documentation.
├── LICENSE #idem.
├── dist/
│   └── spamfilter-0.0.1.tar
│   └── spamfilter-0.0.1-py3-none-any.whl 
├── src/
│   └── spamfilter/
│       ├── __init__.py
│       ├── classifier.py #NaiveBayes classifier.
│       ├── utils.py #Auxiliary functions.
├── tests/
│   ├── test_classifier.py #Tests for NaiveBayes classifier.
│   ├── test_utils.py #Tests for auxiliary functions.
```

## Requirements For Package Uage
- Python 3.7+
- pandas 1.0.3+
- numpy 1.18.4+

## Requirements For Package Creation
Need to install:
- `hatchling`
- `build`
- `wheel`
- `pytest`
- `setuptools`

## License
This project is licensed under the MIT License. See the LICENSE file for details.
