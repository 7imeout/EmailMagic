import os
import json
import meta
import svm
import email

def main():
    """
    Main entry point / top-level execution here
    """
    labels_dict = extract_labels()
    raw_ts_dict = read_training_set()
    
    processed_ts = preprocess_training_set(raw_ts_dict)
    headers = meta.all_labels(processed_ts)
    print (headers)
    # print("Started SVM training")
    # svm.SVMClassifier().train(processed_ts)
    # print("SVM training done")



def train():
    """
    TODO
    """
    pass


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


def preprocess_training_set(raw_ts_dict):
    """
    Iteratively preprocess each eml file and return a list of preprocessed eml dictionaries.
    Each element in the list is on the form {email id: Message object}
    """
    result = []
    for eml_filename, eml in raw_ts_dict.items():
        result.append({eml_filename : email.message_from_string(eml)})
        # meta.d_print(result, source='main/preprocess_training_set')
        # exit()
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

    # meta.d_print(result, source='preprocess_email (end result)')
    return result


def preprocess_eml_content(raw_eml):
    """
    Preprocess the actual content of the eml file.
    Consists of headers (to, from, etc.) and body.
    """
    processed_eml = {}
    meta.d_print(raw_eml)

    # split raw eml format into headers and body
    first_double_newline = raw_eml.index("\n\n")
    header_lines = raw_eml[:first_double_newline].split('\n')
    body = raw_eml[first_double_newline + 2:]

    # preprocess headers into dict
    for line in header_lines:
        if ':' in line:
            first_colon = line.index(':')
            label = line[:first_colon].strip()
            detail = line[first_colon + 1:].strip()
            meta.d_print(label, ':', detail, source='main/process_eml_content (header)')
            processed_eml[label] = detail

    # add body to the dict
    meta.d_print(body, source='main/preprocess_eml_content (body)')
    processed_eml['body'] = body

    return processed_eml


if __name__ == '__main__':
    main()