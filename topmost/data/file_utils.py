import os
import json


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_text(path):
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    with open(path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text.strip() + '\n')


def read_jsonlist(path):
    data = list()
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line))
    return data


def save_jsonlist(list_of_json_objects, path, sort_keys=True):
    with open(path, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def split_text_word(texts):
    texts = [text.split() for text in texts]
    return texts
