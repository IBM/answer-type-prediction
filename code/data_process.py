import torch
from torch.utils.data import TensorDataset

import utils


def get_context_representation(
        text,
        tokenizer,
        max_seq_length,
):
    context_tokens = tokenizer.tokenize(text)

    context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    context_ids = context_ids[:(max_seq_length-4)]

    context_ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[unused1]"]) + context_ids + tokenizer.convert_tokens_to_ids(["[SEP]"])

    padding = [0] * (max_seq_length - len(context_ids))
    context_ids += padding

    assert len(context_ids) == max_seq_length

    return context_ids


def prepare_type_labels(
        data_in,
        num_types,
        name_to_type_id
):
    out_type_labels = [0.0] * num_types

    if data_in["category"] == "resource":
        positive_types_names = data_in["type"]
        assert type(positive_types_names) == list
        positive_types_ids = [name_to_type_id[name] for name in positive_types_names]
        for index in positive_types_ids:
            out_type_labels[index] = 1.0
    return out_type_labels


def prepare_answer_category_label(data_in, answer_category_to_id):
    if data_in["category"] == "resource":
        answer_category_label = answer_category_to_id["resource"]
    else:
        assert len(data_in["type"]) == 1
        assert type(data_in["type"]) == list
        answer_category_label = answer_category_to_id[data_in["type"][0]]
    return answer_category_label


def process_data_1(data, type_id_to_name_file, tokenizer, num_types, max_context_len=64, logger=None):
    # requirements:
    # context_ids, type_labels, answer_category_labels, is_type_resource
    type_id_to_name = utils.read_json(type_id_to_name_file)
    if type(list(type_id_to_name.keys())[0]) is not int:
        type_id_to_name = {int(key): value for key, value in type_id_to_name.items()}

    name_to_type_id = utils.exchange_keys_and_values(type_id_to_name)

    context_ids = []
    type_labels = []
    answer_category_labels = []
    is_type_resource = []
    for data_item in data:
        cntxt_ids = get_context_representation(data_item["question"], tokenizer, max_context_len)
        t_labels = prepare_type_labels(data_item, num_types, name_to_type_id)
        a_c_labels = prepare_answer_category_label(data_item, utils.answer_category_to_id)
        is_resource = 1.0 if data_item["category"] == "resource" else 0.0

        context_ids.append(cntxt_ids)
        type_labels.append(t_labels)
        answer_category_labels.append(a_c_labels)
        is_type_resource.append(is_resource)

    context_ids = torch.tensor(context_ids)
    type_labels = torch.tensor(type_labels)
    answer_category_labels = torch.tensor(answer_category_labels)
    is_type_resource = torch.tensor(is_type_resource)

    tensor_dataset = TensorDataset(context_ids, type_labels, answer_category_labels, is_type_resource)
    return tensor_dataset


def process_typed_data_factory(model_number, params, samples, tokenizer, logger):
    if model_number in [1]:
        tensor_data = process_data_1(
            data=samples,
            type_id_to_name_file=params["type_id_to_name_file"],
            tokenizer=tokenizer,
            num_types=params["num_types"],
            max_context_len=params["max_context_length"],
            logger=logger
        )
    else:
        assert False, "Unsupported value passed for model_number"

    return tensor_data
