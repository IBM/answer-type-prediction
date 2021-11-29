import os
import sys
import json
import torch
import logging

import numpy as np

from collections import OrderedDict
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME


answer_category_to_id = {"resource": 0, "number": 1, "date": 2, "string": 3, "boolean": 4}
non_res_type_to_answer_category = {"number": "literal", "date": "literal", "string": "literal", "boolean": "boolean"}
answer_category_to_non_res_type = {"literal": ["number", "data", "string"], "boolean": "boolean"}


def read_json(file_path):
    with open(file_path) as in_file:
        data = json.load(in_file)
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as out_file:
        json.dump(data, out_file)


def exchange_keys_and_values(dict_in):
    return {v: k for k, v in dict_in.items()}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = "".join(key.split(".module"))
        new_state_dict[name] = value
    return new_state_dict


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger("Blink")
    logger.setLevel(10)
    return logger


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


