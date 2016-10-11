from sklearn.feature_extraction.text import CountVectorizer
import os
import email
import codecs

test_files = os.listdir("TESTING/")  # lists all the files in TESTING, relative to current position
test_data = []

training_files = os.listdir("TRAINING")
training_data = []

#loads the data from the files. Ignores some weird encodings, but I do not think it matters
#if someone wants to verify that its ok that would be rad
for file_name in test_files:
    data = codecs.open("TESTING/" + file_name, "r", encoding='utf-8', errors='ignore')
    test_data.append(data.read())

for file_name in training_files:
    data = codecs.open("TRAINING/" + file_name, "r", encoding='utf-8', errors='ignore')
    training_data.append(data.read())


test_data
training_data