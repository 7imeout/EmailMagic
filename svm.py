from meta import Classifier
from sklearn.svm import SVC

class SVMClassifier(Classifier):

    def train(self, training_set):
        """
        create vectors with all words of email and label
        Exctract all the words used in the emails
        select features
        train
        """
        all_features = self.all_words(training_set)

        self.classifier = SVC()


    def classify(self, email):
        return 1

    def all_words(self, emails):
        res = set()
        for email in emails:
            for word in email.body.split():
                res.add(word)
        return res
