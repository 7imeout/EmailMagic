from meta import Classifier

class NaiveBayesianClassifier(Classifier):

    def train(self, training_set):
        self.classifier = None

    def classify(self, email):
        return 1
