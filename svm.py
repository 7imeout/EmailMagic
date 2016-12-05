from meta import Classifier, d_print
from timeit import default_timer as timer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class SVMClassifier(Classifier):

    def train(self, training_set):
        """
        create vectors with all words of email and label
        Exctract all the words used in the emails
        select features
        train
        """
        all_features = self.all_words(training_set)
        features = []
        labels = []


        
        # The number of emails to use for training
        n_laps = len(training_set)
        start = timer()        
        i = 0
        for name, email_data in training_set.items():
            # Find all used words
            f_dict = {}
            for word in email_data["body"].split():
                f_dict[word] = True

            f_vec = []
            for key in all_features:
                if key in f_dict:
                    f_vec.append(1)
                else:
                    f_vec.append(0)

            labels.append(email_data["label"])
            features.append(f_vec)
            # Abort earlier so we can limit the nr of features
            if i == n_laps:
                break
            else:
                i = i + 1

        end = timer()
        d_print("Pre-processing done, t = " + str(end - start), source="SVM")

        # start = timer()
        # self.feature_selection = SelectKBest(f_classif, k=100)
        # important_features = self.feature_selection.fit_transform(features, labels)
        # end = timer()
        # d_print("Feature selection done, t = " + str(end - start), source="SVM")
        

        start = timer()
        self.classifier = SVC()
        self.classifier.fit(features, labels)
        end = timer()
        d_print("Classifier training done, t = " + str(end - start), source="SVM")
        

    def classify(self, email):
        return 1

    def all_words(self, emails):
        res = set()
        for _, email in emails.items():
            for word in email["body"].split():
                res.add(word)
        return list(res)
