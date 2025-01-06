"""
Module: spamfilter
Description:
    A script that contains all functions to test functionalities
    implemented in util.py

Additional Information:
    - Version: 0.0.1
    - Date: 2025-01-06

Author:
    Aaron Castillo <amc224@imperial.ac.uk>
"""

import unittest
import pandas as pd
from spamfilter.utils import tokenize, document_terms, compute_word_counts


class TestUtils(unittest.TestCase):
    """
    Unit tests for the utility script.
    
    For these tests, we follow two approaches: 
    1. ValueError - all the combinations that raise an error. 
    2. Legit case - we use the right elements and expect a valid result.
    """

    def test_tokenize_not_string(self):
        """
        For function tokenize: Test if raises ValueError when 
        input is not a string.
        """
        
        with self.assertRaisesRegex(ValueError, "Message must be a string."):
            tokenize(123)


    def test_tokenize_whitespace_string(self):
        """
        For function tokenize: Test if raises ValueError when 
        input is a whitespace-only string.
        """
        
        with self.assertRaisesRegex(ValueError, "The string is a whitespace. Please provide a valid string."):
            tokenize("   ")


    def test_tokenize_empty_string(self):
        """
        For function tokenize: Test if raises ValueError when 
        input is an empty string.
        """
        
        with self.assertRaisesRegex(ValueError, "The string is empty. Please provide a valid string."):
            tokenize("")


    def test_document_terms_not_list_of_lists(self):
        """
        For function document_terms: Test if raises ValueError when 
        input is not a list of lists.
        """
        
        with self.assertRaisesRegex(ValueError, "Input must be a list of word lists."):
            document_terms("anything_but_a_list")


    def test_document_terms_inner_elements_not_strings(self):
        """
        For function document_terms: Test if raises ValueError when 
        inner elements are not strings.
        """
        
        with self.assertRaisesRegex(ValueError, "All elements of the inner lists must be strings."):
            document_terms([["token", 123]])


    def test_document_terms_empty_input(self):
        """
        For function document_terms: Test if raises ValueError 
        when input is empty.
        """
        
        with self.assertRaisesRegex(ValueError, "Input is empty. Please provide a valid list."):
            document_terms([])


    def test_compute_word_counts_invalid_dataframe(self):
        """
        For function word_counts: Test if raises ValueError when 
        document_term_dataframe is not a DataFrame.
        """
        with self.assertRaisesRegex(ValueError, "document_term_dataframe must be a pandas DataFrame."):
            compute_word_counts("anything_but_a_dataframe", ["spam", "ham"])


    def test_compute_word_counts_invalid_spam_types(self):
        """
        For function word_counts: Test if raises ValueError when 
        spam_types is invalid.
        """
        
        #We create a dummy dtm.
        doc1 = ["call", "here"]
        dtm = document_terms([doc1])
        
        with self.assertRaisesRegex(ValueError, r"spam_types must be a list of strings \('ham' or 'spam'\)."):
            compute_word_counts(dtm, "anything_but_a_list")


    def test_compute_word_counts_length_mismatch(self):
        """
        For function word_counts: Test if raises ValueError when 
        length of spam_types does not match rows in document_term_dataframe.
        """
        
        #We create a dummy dtm.
        doc1 = ["call", "here"]
        dtm = document_terms([doc1])
        
        with self.assertRaisesRegex(ValueError, "Length of spam_types must match the number of rows in document_term_dataframe."):
            compute_word_counts(dtm, ["spam", "ham", "ham"])


    def test_compute_word_counts_empty_dataframe(self):
        """
        For function word_counts: Test if raises ValueError when 
        document_term_dataframe is empty.
        """
        
        #We create an empty dataframe.
        empty_df = pd.DataFrame()
        
        with self.assertRaisesRegex(ValueError, "document_term_dataframe must be non-empty."):
            compute_word_counts(empty_df, ["spam", "ham"])


    def test_compute_word_counts_empty_spam_types(self):
        """
        For function word_counts: Test if raises ValueError when 
        spam_types is empty.
        """
        
        #We create a dummy dtm.
        doc1 = ["call", "here"]
        dtm = document_terms([doc1])
        
        with self.assertRaisesRegex(ValueError, "spam_types must be non-empty."):
            compute_word_counts(dtm, [])


    def test_legit_case(self):
        """
        Test the functions with a valid scenario.
        """

        #Creating tests for tokenize function.    
        message = "this is a test"
        expected = ["this", "is", "a", "test"]
        
        #Check 1. Test for tokenize function.
        self.assertEqual(tokenize(message), expected)      

        #We create variables for document_term_dataframe function.    
        word_lists = [["call", "here"], ["call", "money"]]
        dtm_expected_columns = ["call", "here", "money"]
        dtm = document_terms(word_lists)
        
        #dtm is:  
        #    call  here  money
        #0     1     1      0
        #1     1     0      1        
        
        #Check 2. Check that the columns match the expected ones.
        self.assertEqual(list(dtm.columns), dtm_expected_columns)
        
        #Check 3. Verify that the values of the first row coincide.
        self.assertEqual(dtm.loc[0, "call"], 1)
        self.assertEqual(dtm.loc[0, "here"], 1)       
        self.assertEqual(dtm.loc[0, "money"], 0)
        
        #Check 3. Verify that the values of the second row coincide.
        self.assertEqual(dtm.loc[1, "call"], 1)
        self.assertEqual(dtm.loc[1, "here"], 0)       
        self.assertEqual(dtm.loc[1, "money"], 1)
        
        #We consider the following data for the word_counts function.
        doc1 = ["call", "here", "win", "prize", "money"]
        doc2 = ["call", "money", "call", "money", "bargain"]
        doc3 = ["call", "here", "information"]
        word_lists = [doc1, doc2, doc3]
        dtm = document_terms(word_lists)
        spam_types = ["spam", "spam", "ham"]
        word_counts = compute_word_counts(dtm, spam_types)
        word_counts_expected_columns = ["call", "here", "win", "prize", "money", "bargain", "information"]
        
        #word_counts is: 
        #        call  here  win  prize  money  bargain  information
        #n_ham      1     1    0      0      0        0            1
        #n_spam     3     1    1      1      3        1            0        
        
        #Check 4. Verify that the indexes of word_counts are ["n_ham", "n_spam"].
        self.assertEqual(list(word_counts.index), ["n_ham", "n_spam"])
       
        #Check 5. Verify that the columns match the expected ones..      
        self.assertEqual(list(word_counts.columns), word_counts_expected_columns)
                
        #Check 6. Make sure that all the values in word_counts match the expected ones.
        self.assertEqual(word_counts.loc["n_ham", "call"], 1)
        self.assertEqual(word_counts.loc["n_spam", "call"], 3)
        self.assertEqual(word_counts.loc["n_ham", "here"], 1)
        self.assertEqual(word_counts.loc["n_spam", "here"], 1)
        self.assertEqual(word_counts.loc["n_ham", "win"], 0)
        self.assertEqual(word_counts.loc["n_spam", "win"], 1)
        self.assertEqual(word_counts.loc["n_ham", "prize"], 0)
        self.assertEqual(word_counts.loc["n_spam", "prize"], 1)
        self.assertEqual(word_counts.loc["n_ham", "money"], 0)
        self.assertEqual(word_counts.loc["n_spam", "money"], 3)
        self.assertEqual(word_counts.loc["n_ham", "bargain"], 0)
        self.assertEqual(word_counts.loc["n_spam", "bargain"], 1)
        self.assertEqual(word_counts.loc["n_ham", "information"], 1)
        self.assertEqual(word_counts.loc["n_spam", "information"], 0)