from meta import Classifier
from sklearn import svm

class SVMClassifier(Classifier):

    def train(self, training_set, labels):
        self.classifier = svm.SVC()
        self.classifier.fit(training_set, labels)

    def classify(self, email):
        email_features = []
        for key in sorted(email.iterkeys()):
            email_features.append(email[key])

        return self.classifier.predict([email])[0]
