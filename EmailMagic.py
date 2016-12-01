import os
import json
import email

from meta import d_print
import naive_bayesian


""" GLOBAL VARIABLES """
header_superset = set()


def main():
    """
    Main entry point / top-level execution here
    """
    labels_dict = extract_labels()
    raw_ts_dict = read_training_set()
    processed_ts = preprocess_training_set(labels_dict, raw_ts_dict)

    # INSTANTIATE YOUR CLASSIFIER AND ADD IT TO THE LIST
    nb = naive_bayesian.NaiveBayesianClassifier()
    classifiers = [nb]

    train(classifiers, processed_ts)


def train(classifiers, training_set):
    """
    Calls training routines of the classifiers
    """
    for cls in classifiers:
        cls.train(training_set)


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
        with open('labels.json') as labels_json:
            return json.load(labels_json)
    else:
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
    training_files = os.listdir("TRAINING")
    training_files_dict = {}
    for file_name in training_files:
        temp = open("TRAINING/" + file_name, "r", encoding='utf-8', errors='ignore')
        training_files_dict[file_name] = temp.read()
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

    # d_print(result, source='preprocess_email (end result)')
    return result


def preprocess_eml_content(raw_eml):
    """
    Preprocess the actual content of the eml file.
    Consists of headers (to, from, etc.) and body.
    """
    global header_superset
    processed_eml = {}

    msg = email.message_from_string(raw_eml)
    for key in msg.keys():
        header_superset.add(key)
        processed_eml[key] = msg.get(key)

    # d_print(processed_eml, source='header dict')

    return processed_eml


if __name__ == '__main__':
    main()