#Import functions and classes from internal modules.
from .utils import tokenize, document_terms, compute_word_counts
from .classifier import NaiveBayes

#Define what gets imported when `from spamfilter import *` is called.
__all__ = ["tokenize", "document_terms", "compute_word_counts", "NaiveBayes"]