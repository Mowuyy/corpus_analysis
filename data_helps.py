# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
import pandas as pd


def corpus_cut(corpus_data, cpu_count):
    corpus_len = len(corpus_data)
    seg_len = int(corpus_len / cpu_count)
    indexes = list(range(corpus_len))
    corpus_list = [corpus_data[(i - 1) * seg_len:i * seg_len] for i in range(1, cpu_count + 1)]
    index_list = [indexes[(i - 1) * seg_len:i * seg_len] for i in range(1, cpu_count + 1)]
    if corpus_len > cpu_count and corpus_len % 2 != 0:
        corpus_list[-1].append(corpus_data[-1])
        index_list[-1].append(indexes[-1])
    elif corpus_len < cpu_count:
        corpus_list = [[item] for item in corpus_data]
        index_list = [[index] for index in indexes]
    return index_list, corpus_list


def min_label_count(tmp_list, semi_label=-1):
    tmp_count_map = Counter(tmp_list)
    del tmp_count_map[semi_label]
    if len(tmp_count_map.keys()) < 2:
        raise ValueError("at least 2 categories")
    min_label_value = min(tmp_count_map.values())
    min_label_key = list(tmp_count_map.keys())[list(tmp_count_map.values()).index(min_label_value)]
    return min_label_key, min_label_value


def get_filter_sample(corpus_file, filename_label_list=None, extract_rules=True):
    df = pd.read_table(corpus_file, sep="\t", header=None, names=["label", "filename", "document"])
    filter_sample = df[np.equal(df.duplicated(subset=["document"]), True)]["filename"].tolist()
    df = df.drop_duplicates(subset=["document"])
    df = df.drop_duplicates(subset=["filename", "label"], keep="last")
    if extract_rules:
        filter_sample = list(set(dict(filename_label_list).keys()) - set(df["filename"]))
    df.to_csv(corpus_file, sep="\t", header=False, index=False)
    return filter_sample


def del_corpus_content_by_filename(corpus_file, filename_label_list):
    df = pd.read_table(corpus_file, sep="\t", header=None, names=["label", "filename", "document"])
    df = df[~df["filename"].isin(f for f, _ in filename_label_list)]
    min_label_num = df.groupby("label").count()["filename"].min()
    if min_label_num < 2:
        raise ValueError("at least 2 categories")
    df.to_csv(corpus_file, sep="\t", header=False, index=False)


class TmpCorpusResult(object):

    __slots__ = ("unlabeled_corpus", "unlabeled_label", "filename")

    def __init__(self, unlabeled_corpus, unlabeled_label, filename):
        self.unlabeled_corpus = unlabeled_corpus
        self.unlabeled_label = unlabeled_label
        self.filename = filename


def read_tmp_corpus(tmp_corpus_file, split_char="\t"):
    label_, f_, content_ = list(), list(), list()
    with open(tmp_corpus_file, "r", encoding="utf-8") as f:
        for row in f.readlines():
            label, f, content = row.strip().split(split_char)
            label_.append(int(label.strip()))
            f_.append(str(f.strip()))
            content_.append(str(content.strip()))
    corpus_result = TmpCorpusResult(content_, label_, f_)
    return corpus_result


def add_corpus_content_and_filename(tmp_corpus_file, old_corpus_file):
    with open(old_corpus_file, "a", encoding="utf-8") as fw:
        with open(tmp_corpus_file, "r", encoding="utf-8") as fr:
            for row in fr.readlines():
                content = row.strip()
                if not content:
                    continue
                fw.write(content + "\n")


def split_corpus(corpus_file, validate_percentage=.2, seed=10):
    data = list()
    with open(corpus_file, "r", encoding="utf-8") as corpus_f:
        for row in corpus_f.readlines():
            data.append(row)
    sample_num = len(data)
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(sample_num))
    split_sample_index = -1 * int(validate_percentage * float(sample_num))
    train_index = shuffle_indices[:split_sample_index]
    validate_index = shuffle_indices[split_sample_index:]
    train_data = list()
    for index in train_index:
        train_data.append(data[index])
    validate_data = list()
    for index in validate_index:
        validate_data.append(data[index])
    return train_data, validate_data


def split_label_and_content(data, split_char="\t"):
    label, content = list(), list()
    for row in data:
        l, _, c = row.split(split_char)
        label.append(int(l.strip()))
        content.append(c.strip())
    return label, content


def split_filename_and_content(data, split_char="\t"):
    filename, content = list(), list()
    for row in data:
        f, c = row.split(split_char)
        filename.append(f)
        content.append(c.strip())
    return filename, content


def merge_content_and_labels(filename_label_map, filename, content, return_filename=False):
    _labels, _filename, _content = list(), list(), list()
    for f, label in filename_label_map.items():
        if f not in filename:
            continue
        index = filename.index(f)
        _labels.append(int(label))
        _content.append(content[index])
        if return_filename:
            _filename.append(f)
    if return_filename:
        return _labels, _filename, _content
    return _labels, _content
