# -*- coding: utf-8 -*-
import subprocess
import os


def base_convert(cmd):
    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return "".join(row.strip() for row in output.decode().split("\n") if row.strip())


def tika_convert(fileanme, jar_path=None, java_path='java'):
    if jar_path is None:
        jar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_tools/tika-app.jar")
    return base_convert((java_path, '-jar', jar_path, '-t', '--encoding=utf-8', fileanme))


def antiword_convert(filename, antiword_path=None):
    if antiword_path is None:
        antiword_path = os.path.abspath(os.path.join(os.path.expanduser("~"), "bin/antiword"))
    return base_convert((antiword_path, '-f', filename))
