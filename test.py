# -*- coding: utf-8 -*-
import os

from corpus_analysis.corpus_handler import Files2Text, CorpusClean, CorpusHandler


def load_data():
    kwargs = {
        "host": "xxx.xx.xx.x",
        "user": "xx",
        "password": "xx",
        "database": "xx",
        "port": 3459
    }
    import pymysql
    import json
    db = pymysql.connect(**kwargs)
    cursor = db.cursor()
    sql = """SELECT t1.filename, t2.cluster_label FROM tb_client_cluster_document as t1, """ \
          """tb_client_cluster_doc_label as t2 WHERE t1.label_id=t2.id AND t1.client_id=101"""
    cursor.execute(sql)
    res = cursor.fetchall()
    cursor.close()
    db.close()
    filename_label_map = dict()
    for filename, label in res:
        filename_label_map[filename] = int(json.loads(label)[0])
    return filename_label_map


class ExampleTest(object):

    def __init__(self, sample_path, corpus_file, filename_label_map, tmp_path):
        self._sample_path = sample_path
        self._corpus_file = corpus_file
        self._filename_label_map = filename_label_map
        self._tmp_path = tmp_path

    def file2text_api(self):
        inst = Files2Text()
        inst.start(self._sample_path, output_file=self._corpus_file)

    def corpus_clean_api(self):
        inst = CorpusClean()
        with open(self._corpus_file) as f:
            for sample in f.readlines():
                sample_content = sample.split("\t")[1]
                sample_content = inst.clean_handler(sample_content)
                print(sample_content)

    def corpus_handler_api(self):
        inst = CorpusHandler()
        train_content, validate_content, train_label, validate_label = inst.get_train_and_validate_corpus_by_extract(
            self._sample_path,
            self._corpus_file,
            self._filename_label_map,
            self._tmp_path
        )
        return train_content, validate_content, train_label, validate_label

    @staticmethod
    def predict_str_api(data):
        inst = CorpusClean()
        res = inst.start(data, None)
        print(res)


if __name__ == '__main__':

    sample_path = "/data/cluster/client_id_101"
    tmp_path = ".tmp/sample_path_101"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    corpus_file = os.path.join(".tmp", "corpus_data_101.txt")
    filename_label_map = load_data()

    tool = ExampleTest(sample_path, corpus_file, filename_label_map, tmp_path)

    tool.file2text_api()
    tool.corpus_clean_api()
    tool.corpus_handler_api()

