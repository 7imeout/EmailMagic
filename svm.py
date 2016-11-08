from meta import Classifier
from sklearn import svm

class SVMClassifier(Classifier):

    def train(self, training_set):
        # self.classifier = svm.SVC()
        # self.classifier.fit()

    def classify(self, email):
        return 1
