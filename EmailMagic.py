import json
import email
import os
from pprint import pprint

from meta import d_print
import naive_bayesian
import decision_tree

""" GLOBAL VARIABLES """
header_superset = set()
header_superset.add('body')


def main():
    """
    Main entry point / top-level execution here
    """
    labels_dict = extract_labels()
    raw_ts_dict = read_training_set()
    processed_ts = preprocess_training_set(labels_dict, raw_ts_dict)

    split = len(processed_ts)//3

    testing = dict(list(processed_ts.items())[:split]) #1/3
    training = dict(list(processed_ts.items())[split:]) #2/3

    print(len(training), len(testing))
    # TODO: INSTANTIATE YOUR CLASSIFIER AND ADD IT TO THE DICT
    nb = naive_bayesian.NaiveBayesianClassifier()
    dt = decision_tree.DecisionTreeClassifier()
    classifiers = {'Naive Bayesian': nb,
                   "Decision Tree": dt}

    train(classifiers, training)
    #classify(classifiers, testing)

def train(classifiers, training_set):
    """
    Calls training routines of the classifiers
    """
    for cls_name in classifiers.keys():
        d_print('Starting training', source=cls_name)
        classifiers[cls_name].train(training_set)
        d_print('Training complete', source=cls_name)


def classify():
    """
    TODO
    """
    pass


def extract_labels():
    """
    Extract labels.txt to build a dictionary and save the result to JSON file.
    If there's already a labels.json available, just read from this file.
    """
    if (os.path.isfile('./labels.json')):
        d_print('Reading labels from the local JSON cache', source='extract_labels')
        with open('labels.json') as labels_json:
            return json.load(labels_json)
    else:
        d_print('Generating a local JSON cache of labels', source='extract_labels')
        labels_txt = open('labels.txt')
        labels_txt_lines = labels_txt.readlines()
        labels_dict = dict((label.split()[1], label.split()[0]) for label in labels_txt_lines)
        with open('labels.json', 'w') as labels_json:
            json.dump(labels_dict, labels_json)
        return labels_dict


def read_training_set():
    """
    Read all raw training files from the directory ./TRAINING into a dictionary, { filename: content ... }
    """
    if (os.path.isfile('./training_set.json')):
        d_print('Reading training set from the local JSON cache', source='read_training_set')
        with open('training_set.json') as training_set_json:
            return json.load(training_set_json)
    else:
        d_print('Generating training set from EML files', source='read_training_set')
        training_files = os.listdir('TRAINING')
        training_files_dict = {}
        for file_name in training_files:
            temp = open('TRAINING/' + file_name, 'r', encoding='utf-8', errors='ignore')
            training_files_dict[file_name] = temp.read()
        with open('training_set.json', 'w') as training_set_json:
            json.dump(training_files_dict, training_set_json)
        return training_files_dict


def preprocess_training_set(labels, raw_ts_dict):
    """
    Iteratively preprocess each eml file and return a list of preprocessed eml dictionaries.
    """
    result = {}

    # pass 1: preprocess with incongruous header set sizes
    for eml_filename, eml in raw_ts_dict.items():
        result[eml_filename] = preprocess_eml(eml_filename, labels[eml_filename], eml)

    # pass 2: fill in missing headers with value of None
    for entry in result.values():
        for header_item in header_superset:
            if header_item not in entry:
                entry[header_item] = None

    return result


def preprocess_eml(eml_filename, label, raw_eml):
    """
    Preprocess a single eml content into a dictionary.
    Add the original filename and the label to the dictionary for identification & training
    """
    result = {'eml_filename': eml_filename, 'label': label}
    content_result = preprocess_eml_content(raw_eml)

    # THIS SYNTAX REQUIRES Python 3.5 !!!
    # merge result and content_result dicts
    result = {**result, **content_result}

    return result


def preprocess_eml_content(raw_eml):
    """
    Preprocess the actual content of the eml file.
    Consists of headers (to, from, etc.) and body.
    """
    global header_superset
    processed_eml = {}

    msg = email.message_from_string(raw_eml)

    # add headers to the dict
    for key in msg.keys():
        header_superset.add(key)
        processed_eml[key] = msg.get(key)

    # add body to the dict
    body_str = parse_multipart_payload_tree(msg.get_payload())
    processed_eml['body'] = body_str

    # d_print(processed_eml, source='header dict')
    return processed_eml


def parse_multipart_payload_tree(payload):
    """
    Annoyingly enough, each payload in our corpus seems to have a different structure.
    Doing a recursive parse here to get all of the payload content concat'd as a string.
    :param payload: root or parent payload; may be a string, list, or a Message object
    :return: single string containing (hopefully) all body content of the given payload
    """
    if isinstance(payload, str):
        # case: simple string / leaf node
        return payload
    elif isinstance(payload, list):
        # case: multiple branches
        children = ''
        for child in payload:
            child_content = parse_multipart_payload_tree(child)
            children += (child_content + '\n') if child_content else ''
        return children
    elif isinstance(payload, email.message.Message):
        # case: Message object / wrapper node
        return parse_multipart_payload_tree(payload.get_payload())
    else:
        # backup case in case I missed anything
        return str(payload)


if __name__ == '__main__':
    main()
