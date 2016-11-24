from meta import Classifier
from sklearn import svm

class SVMClassifier(Classifier):

    def train(self, training_set, label_dict):
        self.classifier = svm.SVC()

        features_train = []
        labels = []
        for email_id in training_set.iterkeys():
            features = []
            labels.append(label_dict[email_id])
            for key in sorted(training_set[email_id].iterkeys()):
                features.append(training_set[key])
            features_train.append(features)

        self.classifier.fit(features_train, labels)

    def classify(self, email):
        email_features = []
        for key in sorted(email.iterkeys()):
            email_features.append(email[key])

        return self.classifier.predict([email])[0]
