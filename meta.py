"""Global flag for DEBUG printing """
DEBUG_MODE = True


def d_print(first_msg, *rest_msg, source='?'):
    """
    Global debug print function; use by importing 'meta' and calling: meta.d_print(...)
    :param first_msg: your debug message, pass in here what you'd pass in to print(...)
    :param rest_msg: variadic argument to support builtin print(...) like param syntax
    :param source: (optional) source of the debug message; function or class name, maybe
    :return: None
    """
    if (DEBUG_MODE):
        end = '\n\n' \
            if '\n' in first_msg or '\n' in rest_msg or (len(str(first_msg)) + len("".join(rest_msg)) > 80) \
            else '\n'
        print("DEBUG from [", source, "]", first_msg, *rest_msg, end=end, flush=True)


class Classifier(object):
    """This is a meta class that serves as a common interface for different classifier classes.

        When you write the classifier classes (see naive_bayesian.py for an example),
        please extend this class and override the methods that this does not have
        the implementation for.

        For unsupervised learning classifications, you'll only need to override classify(...),
        but supervised classifiers are also expected to have train(...) overridden. classify_all(...)
        is already implemented for evaluations; in most cases you won't need to override it.

        Attributes:
            classifier (any): Reference to the classifier object you make/get

        Static Constants:
            N/A

    """
    def __init__(self):
        """
        No-args constructor; sets classifier attr to None.
        """
        self.classifier = None


    def train(self, training_set):  # override me on your supervised-learning classifier class!
        """
        Given a training set, trains the algorithm and initializes the classifier.
        Default implementation is doing nothing, so that the unsupervised learning algorithms can fall back to it.

        The parameter passed in is a dictionary of filename to a dictionary of features.
            Features include: label (1/0), filename, body, and other various header-value pairs.
            Example: { 'TRAIN_12345.eml' : { 'label' : 1,
                                             'eml_filename' : 'TRAIN_12345.eml',
                                             'body' : '... some really long text, html, and css junk ...'
                                             'Subject' : 'some subject',
                                              .
                                              .
                                              .
                                                } }

        :param training_set: training set (see the example above)
        :return: None
        """
        pass


    def classify(self, email):  # override me on your classifier class!
        """
        Given an email (dictionary of email content), classifies it as either ham (1) or spam (0).
        Must be implemented by the subclasses; otherwise will throw NotImplementedError.
        :param email: a dictionary representing an email to be classified
        :return: tuple of (classification result - 1 if ham, 0 if spam, confidence)
        """
        raise NotImplementedError


    def classify_all(self, emails):  # NO NEED to OVERRIDE THIS
        """
        Given a list of emails, or a test set, classifies all of the given emails and
        returns a dictionary of results, in the same format as the 'labels' dictionary.
        :param emails: test set (list of dictionaries representing emails)
        :return: a dictionary with entries like { eml_filename : classification_result }
        """
        return dict((emails['eml_filename'], self.classify(email)) for email in emails)
