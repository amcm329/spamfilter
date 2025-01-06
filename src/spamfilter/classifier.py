"""
Module: spamfilter
Description:
    A script that contains the class NaiveBayes along with its 
    functionalities to perform spam detection and analysis.

Additional Information:
    - Version: 0.0.1
    - Date: 2025-01-06

Author:
    Aaron Castillo <amc224@imperial.ac.uk>
"""

import numpy as np
import pandas as pd
from .utils import tokenize, document_terms, compute_word_counts


class NaiveBayes:
    """
    A Naive Bayes classifier for spam detection.
    """

    def __init__(self, word_counts, spam_types):
        """
        Initializes the classifier with word counts and spam types.

        Params:
             word_counts (pd.DataFrame): A matrix representing word counts for both 
                                         ham and spam classifications.
             spam_types (List (str)): A list indicating the classification of each 
                                      document ("spam" or "ham").
        """
        
        #As part of a defensive programming approach, we check the following 
        #conditions:
        
        #We check that word_counts is a Pandas dataframe.    
        if not isinstance(word_counts, pd.DataFrame):
               raise ValueError("word_counts must be a pandas DataFrame.")

        #We check that spam_types is a list and it contains only "spam" and "ham" labels.  
        if not isinstance(spam_types, list) or not all(label in ["ham", "spam"] for label in spam_types):
              raise ValueError("spam_types must be a list of strings ('ham' or 'spam').")
        
        #We check that word_counts is not empty. 
        if word_counts.empty or word_counts is None:
           raise ValueError("word_counts must be non-empty.")

        #We check that spam_types is not empty.
        if spam_types == [] or spam_types is None:    
           raise ValueError("spam_types must be non-empty.")
           
        #We check that word_counts index is ["n_ham", "n_spam"].
        if list(word_counts.index) != ["n_ham", "n_spam"]:    
           raise ValueError("word_counts index must be [\"n_ham\", \"n_spam\"].")
       
        """
           Private attributes. 
        """
        self.__word_counts = word_counts
        self.__spam_types = spam_types
        
        """
           Public attributes. 
        """
        self.log_probs_ham = []
        self.log_probs_spam = []
        
        
        #NOTE FROM THE DEVELOPER: according to the instructions, the following variables 
        #are lists, but considering the theory behind, we would only have one Pr(S) 
        #and one Pr(H). This idea is supported by the formulas, more accurately, by the 
        #fact that both Pr(S) and Pr(H) are calculated considering the total amount 
        #of messages and they don't depend on any jth word, thus, their values don't
        #vary.
        
        #Therefore, I consider that a list is not the most appropriate 
        #structure to represent these priors, however, I implement them anyway as per 
        #requested in the exercise.
        self.log_prior_ham = []
        self.log_prior_spam = []
        

    def get_spam_types(self):
        """
        Retrieves the private attribute __spam_types.
        
        Returns:
            __spam_types 
        """
        
        return self.__spam_types


    def get_word_counts(self):
        """
        Retrieves the private attribute __word_counts.
        
        Returns:
            __word_counts 
        """
        
        return self.__word_counts


    def fit(self, alpha=0.5):
        """
        Fit the Naive Bayes model by calculating its priors and log probabilities 
        for both ham and spam classifications.

        Params:
            alpha (float): Value that prevents zero estimates.
        """
        
        #As part of a defensive programming approach, we check the following 
        #condition:
        
        #We check that alpha is within the range described in the notes.
        if not(alpha > 0 and alpha <= 1):
           raise ValueError("alpha must be in the range (0,1].")
          
        #We implement the training calculations based on the notes: 
        
        #Step 1: We compute ns and nh (total spam and ham messages).
        ns = self.get_spam_types().count('spam')
        nh = self.get_spam_types().count('ham')
       
        #Step 2: We calculate Pr(S) and Pr(H).
        self.log_prior_spam.append(np.log(ns / (ns + nh)))
        self.log_prior_ham.append(np.log(nh / (ns + nh)))

        #Step 3: We obtain nSj and nHj (number of times the jth word appears in spam and ham messages).
        nSj = np.array(self.get_word_counts().loc["n_spam"].tolist())
        nHj = np.array(self.get_word_counts().loc["n_ham"].tolist())
        
        #Step 4: We get NS and NH (total number of words in ham and spam messages).
        NS = self.get_word_counts().loc["n_spam"].sum()
        NH = self.get_word_counts().loc["n_ham"].sum()

        #Step 5: We calculate Pr(Xj | S) and Pr(Xj | H) with alpha value. 
        self.log_probs_spam = np.log((nSj + alpha) / (NS + alpha * NS + NH))
        self.log_probs_ham = np.log((nHj + alpha) / (NH + alpha * NS + NH))
 

    def classify(self, message):
        """
        Classifies a message as ham or spam.

        Params:
            message (str): The input message to classify.

        Returns:
            str: "ham" or "spam" based on the classification.
        """

        #As part of a defensive programming approach, we check the following 
        #conditions:
        
        #If any of these variables is empty, it means that the fit function hasn't
        #been executed yet.
        if any(x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0) for x in [self.log_probs_ham, self.log_probs_spam, self.log_prior_ham, self.log_prior_spam]):
           raise ValueError("Please, execute fit function first.")
        
        #The message is tokenised.
        #Tests are not needed here as they are inherited
        #from utils.py
        tokens = tokenize(message)
        
        #We get all the available words.
        available_words = list(self.get_word_counts().columns)

        #We create the final scores for both ham and spam.
        final_prob_ham = 0
        final_prob_spam = 0
         
        #In order to create a more efficient code, we store the 
        #mentioned element in local variables so we don't have to 
        #call it every time.
        log_prior_ham = self.log_prior_ham[0]
        log_prior_spam = self.log_prior_spam[0]
        log_probs_ham = self.log_probs_ham
        log_probs_spam = self.log_probs_spam

        #For each one of the tokens, we proceed to calculate their
        #score according to the information provided during the training.
        for token in tokens:
            
            #As per indicated, we skip words that weren't considered during
            #the training.
            if token in available_words:
                index = available_words.index(token)
                final_prob_ham += log_probs_ham[index] + log_prior_ham
                final_prob_spam += log_probs_spam[index] + log_prior_spam 
            
        return "spam" if final_prob_spam > final_prob_ham else "ham"


    def __repr__(self):
        """
        Prints relevant information regarding a NaiveBayes object.
        
        This function is necessary considering the way an object may 
        be called for printing functions:

        >> nb 
        NaiveBayes object
        vocabulary size: 7
        top 5 ham words: call,here,information,win,prize
        top 5 spam words: call,money,here,win,prize
        (prior_ham,prior_spam): (0.3333333,0.6666667)        
        
        In order to achieve such behaviour, we need to override the
        __repr__ function.
       
        Returns:         
            str: A string representation of the NaiveBayes object with specific 
                 information.
        """

        #As part of a defensive programming approach, we check the following 
        #conditions:
        
        #If any of these variables is empty, it means that the fit function hasn't
        #been executed yet.
        if any(x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0) for x in [self.log_probs_ham, self.log_probs_spam, self.log_prior_ham, self.log_prior_spam]):
           raise ValueError("Please, execute fit function first.")
        
        #We obtain all the columns from the word document.
        column_names = list(self.get_word_counts().columns)

        #We get the most relevant ham words in terms of their associated prob.
        sorted_indices_ham = sorted(range(len(self.log_probs_ham)), key=lambda i: self.log_probs_ham[i], reverse=True)
        ordered_columns_ham = [column_names[i] for i in sorted_indices_ham]
      
        #We get the most relevant spam words in terms of their associated prob.
        sorted_indices_spam = sorted(range(len(self.log_probs_spam)), key=lambda i: self.log_probs_spam[i], reverse=True)
        ordered_columns_spam = [column_names[i] for i in sorted_indices_spam]
        
        #Following defensive programming principles, we consider the case then there are less elements
        #in ham and spam words to display the top 5. 
        limit_ham = 5
        limit_spam = 5
        
        additional_info_ham = ""
        additional_info_spam = ""
        
        #In case of existing less than 5 ham words, we carry out an adjustment.
        if len(ordered_columns_ham) < 5: 
           limit_ham = len(ordered_columns_ham) 
           additional_info_ham = "limit for ham words is: ({0}). Adjusting limit...".format(limit_ham)
           
        #In case of existing less than 5 ham words, we carry out an adjustment.
        if len(ordered_columns_spam) < 5: 
           limit_spam = len(ordered_columns_spam) 
           additional_info_spam = "limit for ham words is: ({0}). Adjusting limit...".format(limit_spam)
           
        #We return a set of f-strings that print the current information.
        return (
            "NaiveBayes object\n"
            f"vocabulary size: {self.get_word_counts().shape[1]}\n"
            
            #We print for both ham and spam cases their respective top N of words.
            f"{additional_info_ham}"
            f"top {limit_ham} ham words: {','.join(ordered_columns_ham[0:limit_ham])}\n"
            f"{additional_info_spam}"
            f"top {limit_spam} spam words: {','.join(ordered_columns_spam[0:limit_ham])}\n"
            
            #In this case, since the values are stored as logs, we apply exp to get the original
            #values.
            f"(prior_ham,prior_spam): ({np.exp(self.log_prior_ham[0]):.7f},{np.exp(self.log_prior_spam[0]):.7f})"
        )


    def print(self):
        """
        Prints a NaiveBayes object.
        """
        
        #As indicated before, we override the __repr__ function, so we 
        #just call it.
        print(self.__repr__())


    def compute_confusion_matrix(self, true_labels, predicted_labels, classes = ["ham", "spam"]):
        """
        Creates the confusion matrix. 

        Params:
            true_labels (List(str)): List that contains the actual outputs.
            predicted_labels (List(str)): List that contains the model's predictions.
            classes (List(str)): List with the classes to calculate in the confusion matrix.
            
        Returns:
            dict: a matrix with the counts on "ham" or "spam" based on the expected and predicted values.
        """
        
        #As part of a defensive programming approach, we check the following 
        #conditions:
    
        #Checking that true_labels is not empty.
        if true_labels is None or true_labels == []:
           raise ValueError("true_labels must be non-empty.")

        #Checking that predicted_labels is not empty.
        if predicted_labels is None or predicted_labels == []:
           raise ValueError("predicted_labels must be non-empty.")

        #Checking that the inputs have the same length.
        if len(true_labels) != len(predicted_labels) and len(true_labels) > 0 and len(predicted_labels) > 0:
           raise ValueError("Both true_labels and predicted_labels must have the same length.")

        #Checking that true_labels is a list with "ham" or "spam" values.
        if not isinstance(true_labels, list)or not all(label in ["ham", "spam"] for label in true_labels):
           raise ValueError("true_labels must be a list and contain only \"ham\" or \"spam\" labels.")

        #Checking that predicted_labels is a list with "ham" or "spam" values.        
        if not isinstance(predicted_labels, list) or not all(label in ["ham", "spam"] for label in predicted_labels):
           raise ValueError("predicted_labels must be a list and contain only \"ham\" or \"spam\" labels.")
    
        #We initialize the matrix (dictionary).
        matrix = {}
        
        #For each class, we create an entry in the dictionary. 
        for cls in classes:
            matrix[cls] = {cls2: 0 for cls2 in classes}

        #We fill the numbers according to the four cases. 
        for true, pred in zip(true_labels, predicted_labels):
            matrix[true][pred] += 1
    
        return matrix 
    
    
    def print_confusion_matrix(self, matrix, classes=["ham","spam"]):
        """
        Prints the confusion matrix. 

        Params:
            matrix (dict): Dictionary that contains the counts for both predicted and 
                           expected values.
            classes (List(str)): List with the classes available in the confusion matrix.
        """

        #As part of a defensive programming approach, we check the following 
        #conditions:
 
        #Matrix has to be a dictionary. 
        if not isinstance(matrix, dict):
               raise ValueError("matrix must be a dictionary.")
               
        #Checking that predicted_labels is not empty.
        if matrix is None or matrix == {}:
           raise ValueError("matrix must be not empty.")

        #We check that the values are valid. 
        for row, col_dict in matrix.items():
            
             #Check if all values in a row are zero.
             if all(value == 0 for value in col_dict.values()):
                 raise ValueError(f"All values in row '{row}' are zero.")
        
             #Check if at least one value is negative. 
             for col, value in col_dict.items():
                 if value < 0:
                     raise ValueError(f"Confusion matrix contains at least one negative value.")        
 
        #We print the "Expected / Predicted" headers.
        print(f"Expected \\ {'Predicted':<14}", end="")
        
        #We print the clases.
        for cls in classes:
            print(f"{cls:<10}", end="")
        print(" ")

        #We print the rows.
        for true in classes:
            print(f"{true:<25}", end="")  # Row label (Expected)
            for pred in classes:
                print(f"{matrix[true][pred]:<10}", end="")  # Values
            print(" ")

    
    def compute_accuracy(self, matrix,classes = ["ham", "spam"]):
        """
        Calculate the model's accuracy. 
        
        Params:
            matrix (dict): Dictionary that contains the counts for both predicted and 
                           expected values.
            classes (List(str)): List with the classes available in the confusion matrix.
            
        Returns: 
            float: the model's accuracy.
        """

        #As part of a defensive programming approach, we check the following 
        #conditions:
        
        #Matrix has to be a dictionary.
        if not isinstance(matrix, dict):
               raise ValueError("matrix must be a dictionary.")
               
        #Checking that predicted_labels is not empty.
        if matrix is None or matrix == {}:
           raise ValueError("matrix must be not empty.")

        #We check that the values are valid. 
        for row, col_dict in matrix.items():
            
             #Check if all values in a row are zero.
             if all(value == 0 for value in col_dict.values()):
                 raise ValueError(f"All values in row '{row}' are zero.")
        
             #Check if at least one value is negative. 
             for col, value in col_dict.items():
                 if value < 0:
                     raise ValueError(f"Confusion matrix contains at least one negative value.")        
         
        #We calculate the total number of correct predictions (sum of diagonal elements).
        correct_predictions = sum(matrix[cls][cls] for cls in classes)

        #We compute the total number of samples (sum of all elements in the matrix).
        total_predictions = sum(sum(matrix[row].values()) for row in matrix)

        #We calculate the accuracy.
        accuracy = 0 
        
        if total_predictions > 0:
           accuracy = correct_predictions/total_predictions

        return accuracy