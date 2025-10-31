import os
import random
import sys

from shutil import copyfile
import datetime

import torch

import logging

logger = logging.getLogger()

import numpy as np


def set_global_gpu_env(opt):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    torch.cuda.set_device(opt.gpu)


def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


class TeeLogger:
    """Redirects print() statements to both console and log file"""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a')
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []

    # Main log file for logger statements
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # Console handler (was missing!)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Error handler
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_format)
    logger.addHandler(err_handler)

    logger.setLevel(logging.INFO)

    # IMPORTANT: Also redirect print() statements to log file
    # This captures debug output from model that uses print()
    debug_log_file = os.path.join(output_dir, 'debug.log')
    sys.stdout = TeeLogger(debug_log_file, sys.__stdout__)
    sys.stderr = TeeLogger(debug_log_file, sys.__stderr__)

    logger.info(f"Logging initialized. Logs will be saved to:")
    logger.info(f"  Main log: {output_file}")
    logger.info(f"  Debug log (includes print statements): {debug_log_file}")

    return logger


def get_output_dir(prefix, exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(prefix, 'output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def set_seed(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def setup_output_subdirs(output_dir, *subfolders):
    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    subfolder_list = []
    for sf in subfolders:
        curr_subf = os.path.join(output_subdirs, sf)
        try:
            os.makedirs(curr_subf)
        except OSError:
            pass
        subfolder_list.append(curr_subf)

    return subfolder_list
