from meta import Classifier
from sklearn import svm

class SVMClassifier(Classifier):

    def train(self, training_set):
        self.classifier = svm.SVC()

        features_train = []
        labels = []
        for email in training_set:
            features = []
            labels.append(email["label"])
            for key in sorted(email.keys()):
                if key != "label":
                    features.append(email[key])
            features_train.append(features)

        self.classifier.fit(features_train, labels)

    def classify(self, email):
        email_features = []
        for key in sorted(email.iterkeys()):
            email_features.append(email[key])

        return self.classifier.predict([email])[0]
