from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import codecs

label_lookup = {1:"HAM", 0:"SPAM", "1":"HAM", "0":"SPAM"}
training_files = os.listdir("TRAINING")
data = []

#loads the data from the files. Ignores some weird encodings, but I do not think it matters
#if someone wants to verify that its ok that would be rad
for file_name in training_files:
    temp = codecs.open("TRAINING/" + file_name, "r", encoding='utf-8', errors='ignore')
    data.append(temp.read())


#load the labels
labels = []
for line in open("labels.txt"):
    labels.append(line[0])

#split up the training data and labels into training, testing and validation sets
split = len(data)//3 #where we will split
validation_split = len(data)//10

testing_data = data[validation_split:split] #will be used to test model
testing_labels = labels[validation_split:split]

validation_data = data[:validation_split] #This will be used at the very end to verify results
validation_labels = data[:validation_split]

training_data = data[split:] #2/3
training_labels = labels[split:]

#start tokenizing
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, training_labels)

docs_new = testing_data[0]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
     print('%r => %s' % (doc, label_lookup[category]))





'''
All data from the testing file are unlabeled so we have to split up the training data

test_files = os.listdir("TESTING/")  # lists all the files in TESTING, relative to current position
test_data = []

for file_name in test_files:
    data = codecs.open("TESTING/" + file_name, "r", encoding='utf-8', errors='ignore')
    test_data.append(data.read())

#All data from the testing file are unlabeled so we have to split up the training data
validation_data = test_data[:len(test_data)]
validation_training_labels = training_labels[:len(test_data)]

test_data = test_data[len(test_data):]
test_training_labels = training_labels[len(test_data):]
'''