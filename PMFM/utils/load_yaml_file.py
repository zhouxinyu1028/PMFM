# 加载yaml配置文件

import yaml


def read_yaml(file_path):
    """
        读取yaml配置文件
    :param file_path: yaml文件的路径
    :return:
    """
    with open(file_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)
