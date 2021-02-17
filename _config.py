# -*- coding: utf-8 -*-

# 图像识别API配置
IMAGES_DISCERN_CONF = {
    "app_id": 'xx',
    "api_key": 'xx',
    "secret_key": 'askdla'
}

# 默认日志配置
LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
LOG_DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

# 默认输出文本文件
DEFAULT_OUTPUT_FILE = "./corpus_original_data.txt"

# 文件名与内容默认分隔符
DEFAULT_FILE_CONTENT_SPLIT = "\t"

# 保留词性
RETAIN_POS = {
    "n": "名词",
    "v": "动词",
    "a": "形容词",
    "m": "数量词",
    "q": "量词",
    "r": "代词"
}
