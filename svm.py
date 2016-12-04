from meta import Classifier
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
        n_laps = 200
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
            print(i, name)
            # Abort earlier so we can limit the nr of features
            if i == n_laps:
                break
            else:
                i = i + 1

        important_features = SelectKBest(f_classif, k=100).fit_transform(features, labels)

        self.classifier = SVC()


    def classify(self, email):
        return 1

    def all_words(self, emails):
        res = set()
        for _, email in emails.items():
            for word in email["body"].split():
                res.add(word)
        return list(res)
