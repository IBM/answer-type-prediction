import os
import json
import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler

import utils
import data_process

from biencoder import get_type_model


def read_json(file_path):
    with open(file_path) as in_file:
        data = json.load(in_file)
    return data


def read_training_parameters_file(file_path):
    try:
        with open(file_path, 'r') as fp:
            training_params = json.load(fp)
    except json.decoder.JSONDecodeError:
        with open(file_path) as fp:
            for line in fp:
                line = line.replace("'", "\"")
                line = line.replace("True", "true")
                line = line.replace("False", "false")
                line = line.replace("None", "null")
                training_params = json.loads(line)
                break
    return training_params


class AnswerTypePredictor1:
    def __init__(self, params):
        self.params = params

        if not os.path.exists(self.params["output_path"]):
            os.makedirs(self.params["output_path"])

        self.logger = utils.get_logger(self.params["output_path"])

        self.reranker, self.tokenizer, self.model = self.build_model()

        self.device = self.reranker.device

        self.model.eval()

        self.type_id_to_name = utils.read_json(self.params["type_id_to_name_file"])
        if type(list(self.type_id_to_name.keys())[0]) is not int:
            self.type_id_to_name = {int(key): value for key, value in self.type_id_to_name.items()}

    def build_model(self):
        # Init model
        reranker, tokenizer, model = get_type_model(self.params["type_model"], self.params)
        return reranker, tokenizer, model

    def format_input_data(self, data):
        """
        data is expected to be a list of the following format:
        [
        {"question": },
        {"question": },
        ]
        """

        template = {"id": 0, "question": "", "category": "boolean", "type": ["boolean"]}

        formatted_data = []
        for i, data_item in enumerate(data):
            template_filled = dict(template)  # make a copy of the template before filling it
            template_filled["id"] = i
            template_filled["question"] = data_item["question"]
            formatted_data.append(template_filled)

        return formatted_data

    def get_dataloader(self, data):
        eval_batch_size = self.params["eval_batch_size"]
        eval_tensor_dataset = data_process.process_typed_data_factory(
            model_number=self.params["type_model"],
            params=self.params,
            samples=data,
            tokenizer=self.tokenizer,
            logger=self.logger
        )

        valid_sampler = SequentialSampler(eval_tensor_dataset)
        eval_dataloader = DataLoader(
            eval_tensor_dataset, sampler=valid_sampler, batch_size=eval_batch_size
        )

        return eval_dataloader

    def predict(self, data):
        eval_dataset = self.format_input_data(data)
        eval_dataloader = self.get_dataloader(eval_dataset)

        data_out = []
        formatted_predictions = []

        ans_cat_or_type_id_to_name = utils.exchange_keys_and_values(utils.answer_category_to_id)

        for batch in eval_dataloader:
            context_input, type_label_vec, answer_category_labels, is_type_resource = batch
            with torch.no_grad():
                type_and_cat_predictions = self.reranker.forward_inference(context_input=context_input.to(self.device))
                ans_cat_scores = type_and_cat_predictions["answer_category_prediction_scores"]
                type_scores = type_and_cat_predictions["type_prediction_scores"]

                assert len(ans_cat_scores) == len(type_scores)

                # answer type or category ids
                a_t_or_c_pred_ids = ans_cat_scores.argmax(dim=1).tolist()
                a_t_or_c_pred_names = [ans_cat_or_type_id_to_name[_id] for _id in a_t_or_c_pred_ids]

                type_probs = torch.sigmoid(type_scores)

                k_to_use_for_types = 15

                for i in range(len(type_probs)):
                    if a_t_or_c_pred_names[i] == "resource":
                        t_scores, pred_t_indices = type_probs[i].topk(k_to_use_for_types)
                        pred_t_indices = pred_t_indices.cpu().tolist()
                        ans_type_names = [self.type_id_to_name[_id] for _id in pred_t_indices]
                        ans_cat = "resource"
                    else:
                        ans_type_name = a_t_or_c_pred_names[i]
                        ans_cat = utils.non_res_type_to_answer_category[ans_type_name]
                        ans_type_names = [ans_type_name]

                    formatted_predictions.append({"category": ans_cat, "type": ans_type_names})

        assert len(eval_dataset) == len(formatted_predictions)
        for idx in range(len(eval_dataset)):
            d_out = dict(eval_dataset[idx])
            d_out["category"] = formatted_predictions[idx]["category"]
            d_out["type"] = formatted_predictions[idx]["type"]
            data_out.append(d_out)

        assert len(data_out) == len(eval_dataset)

        return data_out


def get_type_api(config):
    model_specific_training_params_file = os.path.join(config["model_dir"], "training_params.txt")
    model_specific_training_params = read_training_parameters_file(model_specific_training_params_file)

    model_specific_training_params["output_path"] = config["output_directory"]
    model_specific_training_params["hard_negatives_file"] = None
    model_specific_training_params["type_embeddings_path"] = config.get("type_embeddings_path", "")
    model_specific_training_params["ontology_file"] = config.get("ontology_file", "")
    model_specific_training_params["data_path"] = ""
    model_specific_training_params["path_to_model"] = os.path.join(config["model_dir"], "pytorch_model.bin")
    model_specific_training_params["train_batch_size"] = config.get("batch_size", 1)
    model_specific_training_params["eval_batch_size"] = config.get("batch_size", 1)
    model_specific_training_params["data_parallel"] = config.get("data_parallel", False)
    model_specific_training_params["no_cuda"] = config.get("no_cuda", False)
    model_specific_training_params["silent"] = True
    model_specific_training_params["resume_training"] = False
    model_specific_training_params["id_to_type_mapping_file"] = config["id_to_type_mapping_file"]

    params_to_create_the_model_with = model_specific_training_params

    assert "type_model" in params_to_create_the_model_with

    return AnswerTypePredictor1(params_to_create_the_model_with)


if __name__ == "__main__":
    """
    This main block serves as an example of how to use the answer type prediction API

    Example usage:
    python answer_type_prediction_api.py --output_directory ./temp --model_dir ./model --type_embeddings_path ./768_out_of_791_types_pretrained.t7 --ontology_file ./dbpedia_ontology_node_to_parent.json --batch_size 2 --id_to_type_mapping_file ./type_id_to_name.json
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--type_embeddings_path", type=str)
    parser.add_argument("--ontology_file", type=str)
    parser.add_argument("--id_to_type_mapping_file", type=str)
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int)

    config = parser.parse_args()
    config = config.__dict__

    id_to_type_mapping = read_json(config["id_to_type_mapping_file"])

    type_prediction_api = get_type_api(config)

    while True:
        question = input("Question >>> ")

        if len(question) == 0:
            print("Empty strings not allowed")

        data_for_type_prediction = [
            {"question": question}
        ]

        data_out = type_prediction_api.predict(data_for_type_prediction)
        print("Results: {}".format(data_out))

