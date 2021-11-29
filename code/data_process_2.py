import json
import argparse
import math


def read_json(file_path):
    with open(file_path) as in_file:
        data = json.load(in_file)
    return data


def write_json(data, file_path):
    with open(file_path, 'w') as out_file:
        json.dump(data, out_file)


def load_type_hierarchy(filename):
    """Reads the type hierarchy from a TSV file.

    Note: The TSV file is assumed to have a header row.

    Args:
        filename: Name of TSV file.

    Returns:
        A tuple with a types dict and the max hierarchy depth.
    """
#     print('Loading type hierarchy from {}... '.format(filename), end='')
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
#     print('{} types loaded (max depth: {})'.format(len(types), max_depth))
    return types, max_depth


def load_ground_truth(data, type_hierarchy):
    ground_truth = {}
    for question in data:
        if not question['question']:  # Ignoring null questions
#             print('WARNING: question text for ID {} is empty'.format(
#                 question['id']))
            continue
        types = []
        for type in question['type']:
            if question['category'] == 'resource' \
                    and type not in type_hierarchy:
#                 print('WARNING: unknown type "{}"'.format(type))
                continue
            types.append(type)

        ground_truth[question['id']] = {
            'category': question['category'],
            'type': types
        }
#     print('   {} questions loaded'.format(len(ground_truth)))
    return ground_truth


def load_system_output(data):
    system_output = {}
    for answer in data:
        system_output[answer['id']] = {
            'category': answer['category'],
            'type': answer['type']
        }
#     print('   {} predictions loaded'.format(len(system_output)))
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


def sort_types_by_depth(type_list_in, type_to_depth):
    type_list_sorted = sorted(type_list_in, reverse=True, key=lambda x: type_to_depth[x]['depth'])
    return type_list_sorted


def prepare_type_to_parents(type_hierarchy):
    type_to_parents = {}

    for type_name in type_hierarchy:
        parents = []
        current_type_name = type_name
        while current_type_name != "owl:Thing":
            p = type_hierarchy[current_type_name]["parent"]
            if p == "owl:Thing":
                break
            if p not in parents:
                parents.append(p)
            current_type_name = p

        type_to_parents[type_name] = parents

    return type_to_parents


class CompleteTypeLabels:

    @staticmethod
    def complete_gold_labels_0(data_in, type_to_parents, type_to_depth):
        # does not add ancestors. just sorts whatever is there.
        data_out = []
        for data in data_in:
            d_out = dict(data)
            d_out["incomplete_type"] = d_out["type"]
            if d_out["category"] != "resource":
                data_out.append(d_out)
                continue
            completed_types_list = d_out["type"]
            completed_types_list_sorted = sort_types_by_depth(type_list_in=completed_types_list,
                                                              type_to_depth=type_to_depth)
            d_out["type"] = completed_types_list_sorted
            data_out.append(d_out)
        assert len(data_out) == len(data_in)
        return data_out

    @staticmethod
    def complete_gold_labels_1(data_in, type_to_parents, type_to_depth):
        # add all ancestors and sort in decending order of depth
        data_out = []
        for data in data_in:
            d_out = dict(data)
            if d_out["category"] != "resource":
                data_out.append(d_out)
                continue
            completed_types_set = set(d_out["type"])
            for answer_type in d_out["type"]:
                parents = type_to_parents[answer_type]
                completed_types_set.update(parents)
            completed_types_list = list(completed_types_set)
            completed_types_list_sorted = sort_types_by_depth(type_list_in=completed_types_list,
                                                              type_to_depth=type_to_depth)
            d_out["type"] = completed_types_list_sorted
            data_out.append(d_out)
        assert len(data_out) == len(data_in)
        return data_out


if __name__ == "__main__":
    # example usage
    # python data_process_2.py --data_file_in ./train.json --data_file_out ./train_completed_labels.json --type_hierarchy ./smart_dbpedia_types_with_location.tsv

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_in', type=str)
    parser.add_argument('--data_file_out', type=str)
    parser.add_argument('--type_hierarchy', type=str)
    config = parser.parse_args()
    config = config.__dict__

    training_data = read_json(config["data_file_in"])
    type_hierarchy_file = config["type_hierarchy"]

    metrics = evaluate_answer_type_prediction(system_outputs_in=training_data, ground_truth_in=training_data,type_hierarchy_tsv=type_hierarchy_file)
    print("Input data metrics:  {}".format(metrics))

    type_hierarchy, _ = load_type_hierarchy(type_hierarchy_file)
    type_to_parents = prepare_type_to_parents(type_hierarchy)

    training_data_1 = CompleteTypeLabels.complete_gold_labels_1(data_in=training_data, type_to_parents=type_to_parents, type_to_depth=type_hierarchy)
    metrics = evaluate_answer_type_prediction(system_outputs_in=training_data_1, ground_truth_in=training_data,type_hierarchy_tsv=type_hierarchy_file)

    print("Output data metrics:  {}".format(metrics))

    write_json(training_data_1, config["data_file_out"])

    print("Done")



