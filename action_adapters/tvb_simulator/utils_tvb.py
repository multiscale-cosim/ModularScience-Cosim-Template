import os
import shutil
import logging
from common.utils import directory_utils


def create_logger(path, name, log_level):
    """
    create a logger
    it's important for debugging the co-simulation because each modules is in independent processe
    :param path: path of the result
    :param name: name of the logger and for the file
    :param log_level: level of log
    :return: logger
    """
    # Configure logger
    logger = logging.getLogger(name)
    fh = logging.FileHandler(path + '/log/' + name + '.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if log_level == 0:
        fh.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif log_level == 1:
        fh.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    elif log_level == 2:
        fh.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)
    elif log_level == 3:
        fh.setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)
    elif log_level == 4:
        fh.setLevel(logging.CRITICAL)
        logger.setLevel(logging.CRITICAL)
    return logger


def create_folder(path):
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    #     os.mkdir(path)
    # else:
    #     os.mkdir(path)
    directory_utils.safe_makedir(path)
