from meta import Classifier, d_print
from timeit import default_timer as timer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import copy, re, string

class SVMClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.feature_selection = None
        self.all_features = None
        self.number_of_features = 600

    def train(self, training_set):

        self.all_features = self.all_words(training_set)
        features = []
        labels = []

        # The number of emails to use for training
        n_laps = len(training_set)
        start = timer()
        i = 0
        for _, email_data in training_set.items():
            # Find all used words
            f_vec = self.get_feature_vector(email_data)
            labels.append(email_data["label"])
            features.append(f_vec)

            # Abort earlier so we can limit the nr of features
            if i == n_laps:
                break
            else:
                i = i + 1

        end = timer()
        d_print("Pre-processing done, t = " + str(end - start), source="SVM")

        #  Reduce the number of features
        start = timer()
        self.feature_selection = SelectKBest(f_classif, k = self.number_of_features)
        important_features = self.feature_selection.fit_transform(features, labels)
        end = timer()
        d_print("Feature selection done, t = " + str(end - start), source="SVM")


        #  Train the classifier
        start = timer()
        self.classifier = SVC()
        self.classifier.fit(important_features, labels)
        end = timer()
        d_print("Classifier training done, t = " + str(end - start), source="SVM")


    def classify(self, email):
        features = [self.get_feature_vector(email)]

        # Reduce if we had feature reduction
        if self.feature_selection != None:
            features = self.feature_selection.transform(features)

        pred = self.classifier.predict(features)
        return pred

    def classify_all(self, emails):
        start = timer()
        ret = {}
        for key, email in emails.items():
            ret[key] = self.classify(email)[0]

        end = timer()
        d_print("Classification done, t = " + str(end - start), source="SVM")
        return ret

    def all_words(self, emails):
        res = set()
        for _, email in emails.items():
            for feature in self.get_feature_dict(email):
                res.add(feature)
        return list(res)

    def get_feature_dict(self, email):
        # Stolen from mike. ty dude.
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

    def get_feature_vector(self, email):
        f_dict = self.get_feature_dict(email)
        f_vec = []
        for key in self.all_features:
            if key in f_dict:
                f_vec.append(1)
            else:
                f_vec.append(0)
        return f_vec
