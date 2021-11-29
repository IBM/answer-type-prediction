import json
import argparse
import os

import random

random.seed(300)


def read_json(file_path):
    with open(file_path) as in_file:
        data = json.load(in_file)
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as out_file:
        json.dump(data, out_file)


def filter_data(data_in):
    data_out = []
    for data_item in data_in:
        if data_item["question"] is None:
            continue

        if type(data_item["question"]) == str:
            if len(data_item["question"]) < 1:
                continue

        if data_item["category"] is None:
            continue

        if data_item["type"] is None:
            continue

        if len(data_item["type"]) == 0:
            continue

        data_out.append(data_item)

    return data_out


def split(data_in, train_fraction):
    random.shuffle(data_in)

    num_train = int(train_fraction * len(data_in))

    training_set = data_in[:num_train]
    val_set = data_in[num_train:]

    return {"training_set": training_set, "val_set": val_set}


if __name__ == "__main__":
    # example usage
    # python make_train_val_splits.py --data_file ./task1_dbpedia_train.json --train_set_fraction 0.8 --output_dir ./splits

    # returns:
    # Input dataset length: 40621
    # Filtered dataset length: 37084
    # Training set size: 29667
    # Val set size: 7417
    # Done

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--train_set_fraction', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str)
    config = parser.parse_args()
    config = config.__dict__

    raw_dataset = read_json(config["data_file"])
    print("Input dataset length: {}".format(len(raw_dataset)))

    filtered_dataset = filter_data(raw_dataset)
    print("Filtered dataset length: {}".format(len(filtered_dataset)))

    splits = split(data_in=filtered_dataset, train_fraction=config["train_set_fraction"])

    print("Training set size: {}".format(len(splits["training_set"])))
    print("Val set size: {}".format(len(splits["val_set"])))

    write_json(splits["training_set"], os.path.join(config["output_dir"], "train.json"))
    write_json(splits["val_set"], os.path.join(config["output_dir"], "val.json"))

    print("Done")

