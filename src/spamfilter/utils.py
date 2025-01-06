"""
Module: spamfilter
Description:
    A script that contains all functions to cleanse and transform
    tokens into Document Terms.

Additional Information:
    - Version: 0.0.1
    - Date: 2025-01-06

Author:
    Aaron Castillo <amc224@imperial.ac.uk>
"""

import pandas as pd


def tokenize(message):
    """
    Splits a message string tokens without whitespace.

    Params:
        message (str): The input string to tokenize.

    Returns:
        List (str): A list of token strings with no whitespaces inbetween.

    """
 
    #As part of a defensive programming approach, we check the following 
    #conditions:
 
    #We check if the input is string. 
    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    #We validate that the input is not a string with whitespaces only. 
    if message.isspace(): 
        raise ValueError("The string is a whitespace. Please provide a valid string.")    

    #We check that the string is not the empty string.
    if message == "" or message is None: 
        raise ValueError("The string is empty. Please provide a valid string.")    

    #We return the list. Split function without parameter automatically consideres
    #all kinds of whitespaces.
    return message.split()


def document_terms(word_lists):
    """
    Creates a Document Term from a list of word lists.

    Params:
        word_lists List (List (str)): List of word lists (tokens).

    Returns:
        pd.DataFrame: A Pandas DataFrame where the rows correspond to word lists, columns
                      to unique words, and the values represent the counts for each word 
                      in the word lists.
    """

    #As part of a defensive programming approach, we check the following 
    #conditions:
    
    #We eheck that word_lists is a list of lists.
    if not isinstance(word_lists, list) or not all(isinstance(doc, list) for doc in word_lists):
        raise ValueError("Input must be a list of word lists.")

    #We check if all elements in word_lists are strings.
    if not all(isinstance(word, str) for doc in word_lists for word in doc):
       raise ValueError("All elements of the inner lists must be strings.")
    
    #We check if word_lists is not empty.
    if word_lists == [] or word_lists is None:
       raise ValueError("Input is empty. Please provide a valid list.")

    #We obtain all unique words in word_lists.
    #As shown in the instructions, we keep the order of the words considering
    #the first document, then adding only non-repeated words for the next lists. 
    
    #That is because, the order of the words matter. 
    all_unique_words = []
    for doc in word_lists:
        for word in doc:
            if word not in all_unique_words:
                all_unique_words.append(word)
                
    #We create the corresponding Dataframe with columns and rows as unique words.  
    final_df = pd.DataFrame(0, index=range(len(word_lists)), columns=all_unique_words)

    #We fill the respective counts for each word from word_lists (with repeated words).
    for i, doc in enumerate(word_lists):
        for word in doc:
            final_df.loc[i, word] += 1

    #We return the dataframe.
    return final_df


def compute_word_counts(document_term_dataframe, spam_types):
    """
    Creates a matrix with the overall counts for both spam and ham classifications.

    Params:
        document_term_dataframe (pd.DataFrame): Document Term DataFrame.
        spam_types (List (str)): A list of labels ("ham" or "spam") corresponding to 
                                each classification.

    Returns:
        pd.DataFrame: a 2 x p matrix,where p is the length of the vocabulary, the first row 
                      contains the overall counts for words in ham messages and the second
                      row for spam messages.
    """

    #As part of a defensive programming approach, we check the following 
    #conditions:
        
    #We check that the document_term_dataframe is a Pandas dataframe.
    if not isinstance(document_term_dataframe, pd.DataFrame):
        raise ValueError("document_term_dataframe must be a pandas DataFrame.")

    #We check that spam_types is a list and it contains only "spam" and "ham" labels.  
    if not isinstance(spam_types, list) or not all(label in ["ham", "spam"] for label in spam_types):
        raise ValueError("spam_types must be a list of strings ('ham' or 'spam').")

    #We check that the number unique words (columns) in document_term_dataframe, matches the 
    #number of different labels ("ham" or "spam") in spam_type.
    if len(document_term_dataframe) != len(spam_types) and len(document_term_dataframe) > 0 and len(spam_types) > 0:
        raise ValueError("Length of spam_types must match the number of rows in document_term_dataframe.")

    #We check that document_term_dataframe is not empty. 
    if document_term_dataframe.empty or document_term_dataframe is None:
       raise ValueError("document_term_dataframe must be non-empty.")

    #We check that spam_types is not empty.
    if spam_types == [] or spam_types is None:    
       raise ValueError("spam_types must be non-empty.")

    #We create a column with the categories. 
    document_term_dataframe["label"] = spam_types
   
    #We take the individual counts for each category based on the 
    #category previously created. 
    ham_counts = document_term_dataframe[document_term_dataframe["label"] == "ham"].drop(columns="label").sum()
    spam_counts = document_term_dataframe[document_term_dataframe["label"] == "spam"].drop(columns="label").sum()
 
    #We obtain the final dataframe with the necessary index and the counts 
    #per classification.
    result = pd.DataFrame([ham_counts, spam_counts], index=["n_ham", "n_spam"])

    return result