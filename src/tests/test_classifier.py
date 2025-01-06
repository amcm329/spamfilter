"""
Module: spamfilter
Description:
    A script that contains all functions to test functionalities
    about NaiveBayes classifier.

Additional Information:
    - Version: 0.0.1
    - Date: 2025-01-06

Author:
    Aaron Castillo <amc224@imperial.ac.uk>
"""


import unittest
import pandas as pd
from spamfilter.classifier import NaiveBayes
from spamfilter.utils import document_terms, compute_word_counts


class TestNaiveBayes(unittest.TestCase):
    """
    Unit tests for the NaiveBayes classifier.
    
    For these tests, we follow two approaches: 
    1. ValueError - all the combinations that raise an error. 
    2. Legit case - we use the right elements and expect a valid result.
    """

    def setUp(self):
        """
        Initialize test data and objects for the tests.
        """
        
        #We define examples taken from the notes.
        doc1 = ["call", "here", "win", "prize", "money"]
        doc2 = ["call", "money", "call", "money", "bargain"]
        doc3 = ["call", "here", "information"]
        word_lists = [doc1, doc2, doc3]

        #We create document_term_dataframe and word_counts
        dtm = document_terms(word_lists)
        self.spam_types = ["spam", "spam", "ham"]
        self.word_counts = compute_word_counts(dtm, self.spam_types)
        
        #We initialize a NaiveBayes object.
        self.nb = NaiveBayes(self.word_counts, self.spam_types)


    def test_invalid_word_counts_type(self):
        """
        For NaiveBayes object: Test if ValueError is raised with the correct message when 
        word_counts is not a DataFrame.
        """
        
        with self.assertRaisesRegex(ValueError, "word_counts must be a pandas DataFrame."):
            NaiveBayes([], self.spam_types)


    def test_invalid_spam_types_type(self):
        """
        For NaiveBayes object: Test if ValueError is raised with the correct message when 
        spam_types is not a valid list.
        """
        
        with self.assertRaisesRegex(ValueError, r"spam_types must be a list of strings \('ham' or 'spam'\)."):
            NaiveBayes(self.word_counts, "anything_but_a_list")


    def test_empty_word_counts(self):
        """
        For NaiveBayes object: Test if ValueError is raised with the correct message when 
        word_counts is empty.
        """
        
        empty_df = pd.DataFrame()
        
        with self.assertRaisesRegex(ValueError, "word_counts must be non-empty."):
            NaiveBayes(empty_df, self.spam_types)


    def test_empty_spam_types(self):
        """
        For NaiveBayes object: Test if ValueError is raised with the correct message when 
        spam_types is empty.
        """
        
        with self.assertRaisesRegex(ValueError, "spam_types must be non-empty."):
            NaiveBayes(self.word_counts, [])


    def test_invalid_word_counts_index(self):
        """
        For NaiveBayes object: Test if ValueError is raised with the correct message when 
        word_counts index is incorrect.
        """
        
        #We create another word_counts and modify its index.
        invalid_word_counts = self.word_counts.copy()
        invalid_word_counts.index = ["n_jam", "n_charm"]
        
        with self.assertRaisesRegex(ValueError, r"word_counts index must be \[\"n_ham\", \"n_spam\"\]."):
            NaiveBayes(invalid_word_counts, self.spam_types)


    def test_invalid_alpha_in_fit(self):
        """
        For fit function: Test if ValueError is raised with the correct message when 
        alpha is NOT in the range (0,1].
        """
        
        with self.assertRaisesRegex(ValueError, r"alpha must be in the range \(0,1\]."):
            self.nb.fit(alpha=-0.1)


    def test_print_without_fit(self):
        """
        For print function: Test if ValueError is raised with the correct message when 
        print function is called before fit function.
        """
        
        with self.assertRaisesRegex(ValueError, "Please, execute fit function first."):
            self.nb.print()


    def test_classify_without_fit(self):
        """
        For classify function: Test if ValueError is raised with the correct message when 
        classify function is called before fit function.
        """
        
        with self.assertRaisesRegex(ValueError, "Please, execute fit function first."):
            self.nb.classify("this is a message")


    def test_invalid_confusion_matrix_inputs_1(self):
        """
        For compute_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid inputs (length mismatch).
        """
        with self.assertRaisesRegex(ValueError, "Both true_labels and predicted_labels must have the same length."):
             self.nb.compute_confusion_matrix(["spam","spam", "ham"], ["spam", "ham"])


    def test_invalid_confusion_matrix_inputs_2(self):
        """
        For compute_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid inputs (true_labels is empty).
        """
        with self.assertRaisesRegex(ValueError, "true_labels must be non-empty."):
            self.nb.compute_confusion_matrix([], ["spam", "ham"])
 
 
    def test_invalid_confusion_matrix_inputs_3(self):
        """
        For compute_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid inputs (predicted_labels is empty).
        """
        with self.assertRaisesRegex(ValueError, "predicted_labels must be non-empty."):
            self.nb.compute_confusion_matrix(["spam", "ham"], [])


    def test_invalid_confusion_matrix_inputs_4(self):
        """
        For compute_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid inputs (true_labels is not a valid structure).
        """
     
        with self.assertRaisesRegex(ValueError, r"true_labels must be a list and contain only \"ham\" or \"spam\" labels."):
             self.nb.compute_confusion_matrix(["invalid", "label"],["spam", "ham"])


    def test_invalid_confusion_matrix_inputs_5(self):
        """
        For compute_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid inputs (predicted_labels is not a valid structure).
        """
        
        with self.assertRaisesRegex(ValueError, r"predicted_labels must be a list and contain only \"ham\" or \"spam\" labels."):
            self.nb.compute_confusion_matrix(["spam", "ham"], ["invalid", "label"])


    def test_invalid_print_confusion_matrix_1(self):
        """
        For print_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid confusion matrix structure (one of the rows is all zeroes).
        """
        with self.assertRaisesRegex(ValueError, "All values in row 'ham' are zero."):
            self.nb.print_confusion_matrix(matrix={"ham": {"spam": 0, "ham": 0}})


    def test_invalid_print_confusion_matrix_2(self):
        """
        For compute_confusion_matrix function: Test if ValueError is raised with the correct 
        message for invalid confusion matrix structure (one of the values is negative).
        """
        with self.assertRaisesRegex(ValueError, "Confusion matrix contains at least one negative value."):
            self.nb.print_confusion_matrix(matrix={"ham": {"spam": 5, "ham": -1}})


    def test_invalid_compute_accuracy_1(self):
        """
        For compute_accuracy function: Test if ValueError is raised with the correct 
        message for invalid confusion matrix structure (matrix is not a dictionary).
        """
        with self.assertRaisesRegex(ValueError, "matrix must be a dictionary."):
             self.nb.compute_accuracy([])
        
        
    def test_invalid_compute_accuracy_2(self):
        """
        For compute_accuracy function: Test if ValueError is raised with the correct 
        message for invalid confusion matrix structure (matrix is empty).
        """
        with self.assertRaisesRegex(ValueError, "matrix must be not empty."):
             self.nb.compute_accuracy({})
        
       
    def test_invalid_compute_accuracy_3(self):
        """
        For compute_accuracy function: Test if ValueError is raised with the correct 
        message for invalid confusion matrix structure (one of the rows is all zeroes).
        """
        
        with self.assertRaisesRegex(ValueError, "All values in row 'ham' are zero."):
             self.nb.compute_accuracy(matrix={"ham": {"spam": 0, "ham": 0}})
 

    def test_invalid_compute_accuracy_4(self):
        """
        For compute_accuracy function: Test if ValueError is raised with the correct 
        message for invalid confusion matrix structure (one of the values is negative).
        """
        with self.assertRaisesRegex(ValueError, "Confusion matrix contains at least one negative value."):
             self.nb.compute_accuracy(matrix={"ham": {"spam": 5, "ham": -1}})


    def test_legit_case(self):
        """
        Test the classifier in a valid scenario by using the previous example data.
        """
        
        #We fit the classifier.
        self.nb.fit(alpha=1)

        #We classify a test message (result is spam in this case).
        test_message = "call money information"
        result = self.nb.classify(test_message)
        
        #Check 1. Verify that classify returns a valid label. 
        self.assertEqual(result, "spam")

        #We create a customised confusion matrix and calculate its accuracy.
        true_labels = ["spam", "spam", "ham"]
        predicted_labels = [self.nb.classify(message) for message in ["call here", "money bargain", "information here"]]
        confusion_matrix = self.nb.compute_confusion_matrix(true_labels, predicted_labels)
        
        #Accuracy is 1.
        accuracy = self.nb.compute_accuracy(confusion_matrix)

        #Check 2. We validate the confusion matrix structure.
        self.assertEqual(set(confusion_matrix.keys()), {"ham", "spam"})
        
        #Check 3. We verify that the confusion matrix only contains ham and spam labels.
        self.assertTrue(all(cls in confusion_matrix["ham"] for cls in ["ham", "spam"]))
        self.assertTrue(all(cls in confusion_matrix["spam"] for cls in ["ham", "spam"]))

        #Check 4. We check that all values in the matrix are non-negative.
        self.assertEqual(all(value >= 0 for row in confusion_matrix.values() for value in row.values()), True)

        #Check 5. We validate that no row has all its values as 0.
        self.assertEqual(not any(all(value == 0 for value in row.values()) for row in confusion_matrix.values()),True)
    
        #Check 6. We validate that accuracy is 1.
        self.assertEqual(accuracy, 1.0)