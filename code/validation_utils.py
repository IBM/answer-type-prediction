import json
import os
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

import utils
import data_process


def load_type_hierarchy(filename):
    """Reads the type hierarchy from a TSV file.

    Note: The TSV file is assumed to have a header row.

    Args:
        filename: Name of TSV file.

    Returns:
        A tuple with a types dict and the max hierarchy depth.
    """
    print('Loading type hierarchy from {}... '.format(filename), end='')
    types = {}
    max_depth = 0
    with open(filename, 'r') as tsv_file:
        next(tsv_file)  # Skip header row
        for line in tsv_file:
            fields = line.rstrip().split('\t')
            type_name, depth, parent_type = fields[0], int(fields[1]), fields[2]
            types[type_name] = {'parent': parent_type,
                                'depth': depth}
            max_depth = max(depth, max_depth)
    print('{} types loaded (max depth: {})'.format(len(types), max_depth))
    return types, max_depth


def load_ground_truth(data, type_hierarchy):
    ground_truth = {}
    for question in data:
        if not question['question']:  # Ignoring null questions
            print('WARNING: question text for ID {} is empty'.format(
                question['id']))
            continue
        types = []
        for type in question['type']:
            if question['category'] == 'resource' \
                    and type not in type_hierarchy:
                print('WARNING: unknown type "{}"'.format(type))
                continue
            types.append(type)

        ground_truth[question['id']] = {
            'category': question['category'],
            'type': types
        }
    print('   {} questions loaded'.format(len(ground_truth)))
    return ground_truth


def load_system_output(data):
    system_output = {}
    for answer in data:
        system_output[answer['id']] = {
            'category': answer['category'],
            'type': answer['type']
        }
    print('   {} predictions loaded'.format(len(system_output)))
    return system_output


def dcg(gains, k=5):
    """Computes DCG for a given ranking.

    Traditional DCG formula: DCG_k = sum_{i=1}^k gain_i / log_2(i+1).
    """
    dcg = 0
    for i in range(0, min(k, len(gains))):
        dcg += gains[i] / math.log(i + 2, 2)
    return dcg


def ndcg(gains, ideal_gains, k=5):
    """Computes NDCG given gains for a ranking as well as the ideal gains."""
    return dcg(gains, k) / dcg(ideal_gains, k)


def get_type_path(type, type_hierarchy):
    """Gets the type's path in the hierarchy (excluding the root type, like
    owl:Thing).

    The path for each type is computed only once then cached in type_hierarchy,
    to save computation.
    """
    if 'path' not in type_hierarchy[type]:
        type_path = []
        current_type = type
        while current_type in type_hierarchy:
            type_path.append(current_type)
            current_type = type_hierarchy[current_type]['parent']
        type_hierarchy[type]['path'] = type_path
    return type_hierarchy[type]['path']


def get_type_distance(type1, type2, type_hierarchy):
    """Computes the distance between two types in the hierarchy.

    Distance is defined to be the number of steps between them in the hierarchy,
    if they lie on the same path (which is 0 if the two types match), and
    infinity otherwise.
    """
    type1_path = get_type_path(type1, type_hierarchy)
    type2_path = get_type_path(type2, type_hierarchy)
    distance = math.inf
    if type1 in type2_path:
        distance = type2_path.index(type1)
    if type2 in type1_path:
        distance = min(type1_path.index(type2), distance)
    return distance


def get_most_specific_types(types, type_hierarchy):
    """Filters a set of input types to most specific types w.r.t the type
    hierarchy; i.e., super-types are removed."""
    filtered_types = set(types)
    for type in types:
        type_path = get_type_path(type, type_hierarchy)
        for supertype in type_path[1:]:
            if supertype in filtered_types:
                filtered_types.remove(supertype)
    return filtered_types


def get_expanded_types(types, type_hierarchy):
    """Expands a set of types with both more specific and more generic types
    (i.e., all super-types and sub-types)."""
    expanded_types = set()
    for type in types:
        # Adding all supertypes.
        expanded_types.update(get_type_path(type, type_hierarchy))
        # Adding all subtypes (NOTE: this bit could be done more efficiently).
        for type2 in type_hierarchy:
            if type_hierarchy[type2]['depth'] <= type_hierarchy[type]['depth']:
                continue
            type2_path = get_type_path(type2, type_hierarchy)
            if type in type2_path:
                expanded_types.update(type2_path)
    return expanded_types


def compute_type_gains(predicted_types, gold_types, type_hierarchy, max_depth):
    """Computes gains for a ranked list of type predictions.

    Following the definition of Linear gain in (Balog and Neumayer, CIKM'12),
    the gain for a given predicted type is 0 if it is not on the same path with
    any of the gold types, and otherwise it's $1-d(t,t_q)/h$ where $d(t,t_q)$ is
    the distance between the predicted type and the closest matching gold type
    in the type hierarchy and h is the maximum depth of the type hierarchy.

    Args:
        predicted_types: Ranked list of predicted types.
        gold_types: List/set of gold types (i.e., perfect answers).
        type_hierarchy: Dict with type hierarchy.
        max_depth: Maximum depth of the type hierarchy.

    Returns:
        List with gain values corresponding to each item in predicted_types.
    """
    gains = []
    expanded_gold_types = get_expanded_types(gold_types, type_hierarchy)
    for predicted_type in predicted_types:
        if predicted_type in expanded_gold_types:
            # Since not all gold types may lie on the same branch, we take the
            # closest gold type for determining distance.
            min_distance = math.inf
            for gold_type in gold_types:
                min_distance = min(get_type_distance(predicted_type, gold_type,
                                                     type_hierarchy),
                                   min_distance)
            gains.append(1 - min_distance / max_depth)
        else:
            gains.append(0)
    return gains


def evaluate_answer_type_prediction(system_outputs_in, ground_truth_in, type_hierarchy_tsv):
    """Evaluates a system's predicted output against the ground truth.
    """

    # system_output: Dict with the system's predictions.
    # ground_truth: Dict with the ground truth.
    # type_hierarchy: Dict with the type hierarchy.
    # max_depth: Maximum depth of the type hierarchy.
    type_hierarchy, max_depth = load_type_hierarchy(type_hierarchy_tsv)
    ground_truth = load_ground_truth(ground_truth_in, type_hierarchy)
    system_output = load_system_output(system_outputs_in)

    accuracy = []
    ndcg_5, ndcg_10 = [], []
    for question_id, gold in ground_truth.items():
        if question_id not in system_output:
            print('WARNING: no prediction made for question ID {}'.format(
                question_id))
            system_output[question_id] = {}
        predicted_category = system_output[question_id].get('category', None)
        predicted_type = system_output[question_id].get('type', [None])

        if predicted_category != gold['category']:
            accuracy.append(0)
            gains = [0]
            ideal_gains = [1]
        else:
            # Category has been correctly predicted -- proceed to type evaluation.
            accuracy.append(1)

            if gold['category'] == 'boolean' and predicted_category == 'boolean':
                gains = [1]
                ideal_gains = [1]
            elif len(predicted_type) == 0:
                gains = [0]
                ideal_gains = [1]
            elif gold['category'] == 'literal':
                gains = [1 if gold['type'][0] == predicted_type[0] else 0]
                ideal_gains = [1]
            elif gold['category'] == 'resource':
                if len(gold['type']) == 0:
                    print('WARNING: no gold types given for question ID {}'.format(
                        question_id))
                    continue
                # Filters gold types to most specific ones in the hierarchy.
                gold_types = get_most_specific_types(gold['type'], type_hierarchy)

                gains = compute_type_gains(predicted_type, gold_types,
                                           type_hierarchy, max_depth)
                ideal_gains = sorted(
                    compute_type_gains(
                        get_expanded_types(gold_types, type_hierarchy), gold_types,
                        type_hierarchy, max_depth), reverse=True)

            else:
                raise Exception(f"Invalid category: {gold['category']}")

        ndcg_5.append(ndcg(gains, ideal_gains, k=5))
        ndcg_10.append(ndcg(gains, ideal_gains, k=10))

    all_metrics = {
        "num_questions_considered_for_category_prediction": len(accuracy),
        "category_prediction_acc": sum(accuracy) / len(accuracy),
        "num_questions_considered_for_type_ranking": len(ndcg_5),
        "NDCG_at_5": sum(ndcg_5) / len(ndcg_5),
        "NDCG_at_10": sum(ndcg_10) / len(ndcg_10),
    }

    # print('\n')
    # print('Evaluation results:')
    # print('-------------------')
    # print('Category prediction (based on {} questions)'.format(len(accuracy)))
    # print('  Accuracy: {:5.3f}'.format(sum(accuracy) / len(accuracy)))
    # print('Type ranking (based on {} questions)'.format(len(ndcg_5)))
    # print('  NDCG@5:  {:5.3f}'.format(sum(ndcg_5) / len(ndcg_5)))
    # print('  NDCG@10: {:5.3f}'.format(sum(ndcg_10) / len(ndcg_10)))

    return all_metrics


class Evaluator1:
    # Evaluates performance on answer type prediction

    def __init__(self, eval_dataset_path: str, tokenizer, logger, params, device) -> None:
        self.logger = logger
        self.type_labels = None
        self.category_labels = None
        self.eval_dataset_path = eval_dataset_path
        self.eval_data = None
        self.eval_dataloader = None
        self.params = params
        self.device = device
        self.tokenizer = tokenizer

        self.type_id_to_name = utils.read_json(self.params["type_id_to_name_file"])
        if type(list(self.type_id_to_name.keys())[0]) is not int:
            self.type_id_to_name = {int(key): value for key, value in self.type_id_to_name.items()}

        self.are_labels_available = True
        self.create_eval_data_loader()

    # assumption: test data wont have the "category" and "type" fields.
    # so add some dummy values to make the data pre-processing and data loader work
    def add_fake_labels_if_necessary(self, data_in):
        data_out = []
        no_labels_count = 0
        for d in data_in:
            d_out = dict(d)
            if ("category" not in d) or ("type" not in d):
                no_labels_count += 1
                # dont edit data_in
                d_out = dict(d)
                d_out["category"] = "boolean"
                d_out["type"] = ["boolean"]
            data_out.append(d_out)

        if no_labels_count != 0:
            assert no_labels_count == len(data_in), "This code proceeds only when either (a) ALL examples have labels or, (b) When NONE of the examples have labels"
            self.are_labels_available = False

        return data_out

    def create_eval_data_loader(self):
        eval_dataset_raw = utils.read_json(self.eval_dataset_path)
        eval_dataset = self.add_fake_labels_if_necessary(eval_dataset_raw)

        assert len(eval_dataset_raw) == len(eval_dataset)

        self.eval_data = eval_dataset

        eval_batch_size = self.params["eval_batch_size"]
        eval_tensor_dataset = data_process.process_typed_data_factory(
            model_number=self.params["type_model"],
            params=self.params,
            samples=eval_dataset,
            tokenizer=self.tokenizer,
            logger=self.logger
        )

        valid_sampler = SequentialSampler(eval_tensor_dataset)
        self.eval_dataloader = DataLoader(
            eval_tensor_dataset, sampler=valid_sampler, batch_size=eval_batch_size
        )

    def get_predictions(self, model):
        data_out = []
        formatted_predictions = []

        ans_cat_or_type_id_to_name = utils.exchange_keys_and_values(utils.answer_category_to_id)

        for batch in tqdm(self.eval_dataloader, desc="Evaluation"):
            context_input, type_label_vec, answer_category_labels, is_type_resource = batch
            with torch.no_grad():
                type_and_cat_predictions = model.forward_inference(context_input=context_input.to(self.device))
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

        assert len(self.eval_data) == len(formatted_predictions)
        for idx in range(len(self.eval_data)):
            d_out = dict(self.eval_data[idx])
            d_out["category"] = formatted_predictions[idx]["category"]
            d_out["type"] = formatted_predictions[idx]["type"]
            data_out.append(d_out)

        assert len(data_out) == len(self.eval_data)

        return data_out

    def evaluate(self, model, dump=False):
        self.logger.info("--------------- Started evaluation ---------------")

        model.model.eval()

        assert self.eval_dataset_path != ""
        dataset_name = os.path.basename(self.eval_dataset_path)
        self.logger.info("Dataset: {}".format(dataset_name))

        predictions_all = self.get_predictions(model)

        if dump:
            dump_file = os.path.join(self.params["output_path"],  "type_prediction_" + dataset_name)
            utils.write_json(predictions_all, dump_file)
            self.logger.info("Dumped predictions here: {}".format(dump_file))

        if self.are_labels_available is False:
            self.logger.info("Gold labels are not available. Skipping evaluation")
            return

        metrics = evaluate_answer_type_prediction(
            system_outputs_in=predictions_all,
            ground_truth_in=self.eval_data,
            type_hierarchy_tsv=self.params["smart_type_hierarchy_tsv"]
        )

        self.logger.info("Metrics: {}".format(metrics))

        results = {"metrics": metrics}

        # to make the training code use either entity or type accuracy to decide if we have a "new best model"
        # after each epoch
        if self.params["main_metric"] == "cat_acc":
            results["normalized_accuracy"] = metrics["category_prediction_acc"]
        elif self.params["main_metric"] == "ndcg_5":
            results["normalized_accuracy"] = metrics["NDCG_at_5"]
        elif self.params["main_metric"] == "ndcg_10":
            results["normalized_accuracy"] = metrics["NDCG_at_10"]
        else:
            assert False, "Unsupported value for main_metric"

        # put the model to the training mode, in case that is not being done in the training code
        model.model.train()

        self.logger.info("--------------- Completed evaluation ---------------")

        return results


def evaluator_factory(model_number, params, eval_dataset_path, tokenizer, logger, device):
    if model_number in [1]:
        return Evaluator1(
            eval_dataset_path=eval_dataset_path,
            tokenizer=tokenizer,
            logger=logger,
            params=params,
            device=device
        )
    else:
        assert False, "Unsupported model_number"
