from meta import Classifier
from sklearn import tree, cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import copy, re, string
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class DecisionTreeClassifier(Classifier):
    def train(self, training_set):
        self.classifier = None
        labels_arr = []
        features_arr = []

        all_emails = [(self.get_features(training_set[eml]), False if training_set[eml]['label'] == '1' else True) for eml in training_set.keys()]

        for features, label in all_emails:
            labels_arr.append(label)
            features_arr.append(features)

        features_train, features_test, labels_train, labels_test \
            = cross_validation.train_test_split(features_arr, labels_arr, test_size=0.3)

        vectorizer = DictVectorizer()
        features_train_transformed = vectorizer.fit_transform(features_train)
        features_test_transformed = vectorizer.transform(features_test)

        selector = SelectPercentile(f_classif, percentile=1)
        selector.fit(features_train_transformed, labels_train)
        features_train_transformed = selector.transform(features_train_transformed).toarray()
        features_test_transformed  = selector.transform(features_test_transformed).toarray()

        self.classifier = tree.DecisionTreeClassifier()
        cls = self.classifier.fit(features_train_transformed, labels_train)

        importances = cls.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(20):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


        score = cls.score(features_test_transformed, labels_test)
        print(score)

    def classify(self, email):
        pass

    def get_features(self, email):
        features = {}
        body = email['body']

        pattern = re.compile("[a-zA-Z0-9_-]{1,20}")
        for item in body.split(' '):
            if pattern.match(item):
                item = item.lower()
                item = item.translate(str.maketrans('', '', string.punctuation))
                features['BODY__' + item] = True

        email_copy = copy.deepcopy(email)
        email_copy.pop('body', None)
        email_copy.pop('label', None)
        email_copy.pop('eml_filename', None)

        for key in email_copy.keys():
            val = email_copy[key]
            email_copy[key] = True if val else False

        return {**email_copy, **features}