"""
Description:
    A script that contains all implementations and final tests
    by using spamfilter package.

Additional Information:
    - Version: 0.0.1
    - Date: 2025-01-06

Author:
    Aaron Castillo <amc224@imperial.ac.uk>
"""

#Libraries to test our package.
import numpy as np 
import pandas as pd
from spamfilter import tokenize, document_terms, compute_word_counts, NaiveBayes

"""
*****************************************************
*****************************************************
******************* Section D(i) ********************
*****************************************************
*****************************************************

Load train.csv, createa and print a NaiveBayes object.
"""

train_file = "train.csv"
train_data = pd.read_csv(train_file, skip_blank_lines=True)

#As part of defensive programming approach, we remove
#non-string cases like NAN's or empty strings. 
train_data = train_data.dropna(subset=['text'])
train_data = train_data.dropna(subset=['spam_type'])
train_data = train_data[~(train_data == '').all(axis=1)]

#We extract both texts and spam types.
word_lists = []
train_messages = train_data["text"].to_list()
spam_types = train_data["spam_type"].to_list()

#We create word_lists.
for message in train_messages:
    word_lists.append(tokenize(message))

#We create both document term dataframe and word_counts.    
dtm = document_terms(word_lists)
word_counts = compute_word_counts(dtm, spam_types)

#We create and fit NaiveBayes classifier.
nb = NaiveBayes(word_counts, spam_types)
nb.fit(alpha=1)

print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("Section D(i)\n")

#We print the classifier object.
print(nb)
print(" ")

"""
*****************************************************
*****************************************************
******************* Section D(ii) *******************
*****************************************************
*****************************************************

Load test.csv, classify messages for both training and
testing parts. 

In addition, both confusion matrices and accuracies are
printed. 

Finally, there's a comment on the exercise.
"""

#For this section, we use the functions in the clase
#NaiveBayes that calculate both the confusion matrix 
#and the accuracy. 

#Load test.csv
test_file = "test.csv"
test_data = pd.read_csv(test_file, skip_blank_lines=True)

#As part of defensive programming approach, we remove
#non-string cases like NAN's or empty strings. 
test_data = test_data.dropna(subset=['text'])
test_data = test_data.dropna(subset=['spam_type'])
test_data = test_data[~(test_data == '').all(axis=1)]

#We extract both texts and spam types.
test_messages = test_data["text"].to_list()
test_true = test_data["spam_type"].to_list()

#We classify messages in train.csv and test.csv
train_predictions = [nb.classify(message) for message in train_messages]
test_predictions = [nb.classify(message) for message in test_messages]

#We calculate both train and test confusion matrices .
train_confusion_matrix = nb.compute_confusion_matrix(spam_types, train_predictions)
test_confusion_matrix = nb.compute_confusion_matrix(test_true, test_predictions)

#We calculate both train and test accuracies.
train_accuracy = nb.compute_accuracy(train_confusion_matrix)
test_accuracy = nb.compute_accuracy(test_confusion_matrix)

print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("Section D(ii)\n")

# Print confusion matrices and accuracies
print("Train Confusion Matrix:")
nb.print_confusion_matrix(train_confusion_matrix)

print(" ")

print("Test Confusion Matrix:")
nb.print_confusion_matrix(test_confusion_matrix)

print(" ")     

print("Train Accuracy:", train_accuracy)

print(" ")

print("Test Accuracy:", test_accuracy)

print(" ")   

comment_dii = """Comments on this exercise: 

In general, we can see that the test accuracy is slightly better than the train accuracy, but 
here we need to mention the relevance of alpha = 1. 

Basically, by choosing alpha = 1, we are indicating our model to perform UNDERFITTING, that is, 
that the model don't learn from the data. 

That is why it makes sense that the values are somewhat similar, since the model isn't learning
at all, it's just repeating patterns. 

Regarding the confusion matrix, the only thing to enhance has to do with false negatives 
(predicted HAM but expected SPAM). 

The closer alpha to 0, the "better" results we might obtain as long as we are careful on not causing
OVERFITTING.
"""
print(comment_dii)

print(" ")   
 
"""
*****************************************************
*****************************************************
******************* Section D(iii) ******************
*****************************************************
*****************************************************

Tune alpha for highest accuracy and comment on the 
exercise.
"""

#We define alpha values

#By definition, alpha can't be 0, thus, we start 
#with a value that is almost zero.
alphas = np.linspace(0.001, 1, 10)
best_alpha = -1
best_test_accuracy = 0

nb_diii = NaiveBayes(word_counts, spam_types)

print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("Section D(iii)\n")

#We tune alpha according to the instructions.
for current_alpha in alphas:
    nb_diii.fit(alpha=current_alpha)
    test_predictions = [nb_diii.classify(message) for message in test_messages]
    test_confusion_matrix = nb.compute_confusion_matrix(test_true, test_predictions)
    test_accuracy = nb.compute_accuracy(test_confusion_matrix)

    print(f"   Current alpha: {current_alpha}. Current accuracy: {test_accuracy}\n")
    print("    Current confusion matrix: ")
    nb.print_confusion_matrix(test_confusion_matrix)
    print(" ")
    print(" ")
    
    if test_accuracy > best_test_accuracy:
       best_test_accuracy = test_accuracy
       best_alpha = current_alpha

#Print best alpha, accuracy and confustion matrix.
print(" ")
print("--------------------------------------------------------------")
print(" ")
print(f"Best alpha: {best_alpha}\n")
print(f"Best Test Accuracy: {best_test_accuracy}\n")

#Getting predictions for best alpha.
nb_diii.fit(alpha=best_alpha)
final_test_predictions = [nb_diii.classify(message) for message in test_messages]
final_test_confusion_matrix = nb.compute_confusion_matrix(test_true, final_test_predictions)
print("    Best confusion matrix: ")
nb.print_confusion_matrix(final_test_confusion_matrix)

print(" ")

comment_diii = """Comments on this exercise: 

Even though that, generally speaking, a parameter alpha close to 0 is beneficial to the model according to the information printed
and sites like this: 

     https://stackoverflow.com/questions/52319703/naive-bayes-accuracy-increasing-as-increasing-in-the-alpha-value

it can cause OVERFITTING if this is not properly controled. 

Furthermore, the selection of the hyperparameter alpha in this case IS NOT THE MOST REASONABLE because we are using the TESTING SET
to perform this activity.

According to sites like this one: 

      https://stats.stackexchange.com/questions/611659/is-it-a-bad-practice-to-learn-hyperparameters-from-the-training-data-set
      
Hyperparmeter tuning must be performed in the TRAINING SET, otherwise what we are achieving is "fitting" the "unseen data" that
can provoque an OVERFITTING.

Regarding the confusion matrix, the only thing to enhance has to do with false negatives (predicted HAM but expected SPAM). 

To summarise, if we get alpha close to 0 IN THE TRAINING SET is a reasonable idea, otherwise, it is not. 
"""
print(comment_diii)