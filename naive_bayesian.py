import nltk, copy, string, re
from meta import Classifier, d_print

# use 1/3 split
TEST_RATIO = 3

class NaiveBayesianClassifier(Classifier):

    def train(self, training_set):

        all_emails = [(self.get_features(training_set[eml]), False if training_set[eml]['label'] == '1' else True) for eml in training_set.keys()]
        split_point = len(all_emails) // TEST_RATIO
        test_split, train_split = all_emails[:split_point], all_emails[split_point:]

        self.classifier = nltk.NaiveBayesClassifier.train(train_split)

        print()
        self.classifier.show_most_informative_features(100)
        print('\nAccuracy: {:6.4f}%'.format(nltk.classify.accuracy(self.classifier, test_split) * 100))

    def classify(self, email):
        pass

    def get_features(self, email):
        features = {}
        body = email['body']
        ptrn = re.compile("[a-zA-Z0-9_-]{1,20}")

        for token in body.split(' '):
            if ptrn.match(token):
                token = token.lower()
                token = token.translate(str.maketrans('', '', string.punctuation))
                features['BODY__' + token] = True

        email_copy = copy.deepcopy(email)
        email_copy.pop('body', None)
        email_copy.pop('eml_filename', None)
        email_copy.pop('label', None)

        for key in email_copy.keys():
            val = email_copy[key]
            email_copy[key] = True if val else False

        return {**email_copy, **features}
