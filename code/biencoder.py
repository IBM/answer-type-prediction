# This file is adapted from https://github.com/facebookresearch/BLINK

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from ranker_base import BertEncoder, get_model_obj


def safe_divide(numerator, denominator):
    if denominator == 0:
        denominator = 1

    return numerator / denominator


def get_aux_task_weight(training_progress, gamma=10):
    return (2 / (1 + (math.e ** (-gamma * training_progress)))) - 1


class GradientThrottle(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return GradientThrottle.scale * grad_output


def grad_throttle(x, scale=1.0):
    GradientThrottle.scale = scale
    return GradientThrottle.apply(x)


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


class TypedBiEncoderModule1(torch.nn.Module):
    def __init__(self, params):
        super(TypedBiEncoderModule1, self).__init__()
        # bert_model, output_dim, rep_for_ans_cat_pred

        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder(
            bert_model=ctxt_bert,
            output_dim=params["out_dim"],
            rep_for_ans_cat_pred=params["rep_for_ans_cat_pred"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )

        self.config = ctxt_bert.config

        if params["type_embeddings_path"] == "":
            self.additional_type_embedding_layer = nn.Embedding(
                num_embeddings=params["num_types"],
                embedding_dim=params["type_embedding_dim"],
            )
        else:
            type_embedding_weights = torch.load(params["type_embeddings_path"])
            self.additional_type_embedding_layer = nn.Embedding.from_pretrained(
                embeddings=type_embedding_weights,
                freeze=params["freeze_type_embeddings"],
            )

        self.additional_up_project_linear = None
        if not params["no_linear_after_type_embeddings"]:
            # self.additional_up_project_linear = nn.Linear(
            #     in_features=params["type_embedding_dim"],
            #     out_features=self.config.hidden_size,
            #     bias=True,
            # )

            self.additional_up_project_linear = nn.Sequential(
                nn.Linear(
                    in_features=params["type_embedding_dim"],
                    out_features=int(self.config.hidden_size / 3),
                    bias=True,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=int(self.config.hidden_size / 3),
                    out_features=self.config.hidden_size,
                    bias=True,
                ),
            )

            # self.additional_up_project_linear = nn.Sequential(
            #     nn.Linear(
            #         in_features=params["type_embedding_dim"],
            #         out_features=self.config.hidden_size,
            #         bias=True,
            #     ),
            #     nn.GELU(),
            #     nn.LayerNorm(self.config.hidden_size),
            # )

        self.additional_answer_category_pred = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=params["num_answer_categories"],
                    bias=True,
                )

        self.params = params

    def get_type_vectors(self):
        type_vectors = self.additional_type_embedding_layer.weight
        if self.additional_up_project_linear is not None:
            type_vectors = self.additional_up_project_linear(type_vectors)
        return type_vectors

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt
    ):
        if token_idx_ctxt is not None:
            if self.params["freeze_context_bert"]:
                with torch.no_grad():
                    rep_for_type_prediction, rep_for_answer_category_prediction = self.context_encoder(
                        token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                    )
            else:
                rep_for_type_prediction, rep_for_answer_category_prediction = self.context_encoder(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                )

        return rep_for_type_prediction, rep_for_answer_category_prediction


class TypedBiEncoderRanker1(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypedBiEncoderRanker1, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        # if model_path is not None:
        #     self.load_model(model_path)

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.summary_writer = None

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = TypedBiEncoderModule1(self.params)

    # during training, call this function each time before you use the type vectors
    def get_type_vectors(self):
        if self.data_parallel:
            return self.model.module.get_type_vectors()
        else:
            return self.model.get_type_vectors()

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(output_dir, "config.json")
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        rep_for_type_prediction, rep_for_answer_category_prediction = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return rep_for_type_prediction.cpu().detach(), rep_for_answer_category_prediction.cpu().detach()

    def score_candidate(self, text_vecs):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        rep_for_type_prediction, rep_for_answer_category_prediction = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt
        )

        type_vectors = self.get_type_vectors()

        type_prediction_scores = rep_for_type_prediction.mm(type_vectors.t())

        if self.params["data_parallel"]:
            answer_category_prediction_scores = self.model.module.additional_answer_category_pred(rep_for_answer_category_prediction)
        else:
            answer_category_prediction_scores = self.model.additional_answer_category_pred(
                rep_for_answer_category_prediction)

        return type_prediction_scores, answer_category_prediction_scores

    def forward(
            self, context_input, type_labels=None, answer_category_labels=None, is_type_resource=None, iteration_number=None, training_progress=1.0
    ):

        type_prediction_scores, answer_category_prediction_scores = self.score_candidate(
            text_vecs=context_input
        )

        loss = None

        if type_labels is not None:

            answer_category_prediction_loss = F.cross_entropy(answer_category_prediction_scores,  answer_category_labels, reduction="mean")

            bce_loss_function = nn.BCEWithLogitsLoss(reduction="none")

            type_loss_unreduced = bce_loss_function(type_prediction_scores, type_labels.float())

            type_loss_unreduced = type_loss_unreduced * is_type_resource.unsqueeze(dim=1)

            # average loss for all positive types in the batch
            type_loss_positives = safe_divide(
                (type_loss_unreduced * type_labels).sum(), type_labels.sum()
            )

            # average loss for all negative types in the batch
            type_loss_negatives = safe_divide(
                (type_loss_unreduced * (1 - type_labels)).sum(), (1 - type_labels).sum()
            )

            type_loss = (
                                self.params["type_loss_weight_positive"] * type_loss_positives
                        ) + (self.params["type_loss_weight_negative"] * type_loss_negatives)

            if self.params["type_task_importance_scheduling"] == "loss_weight":
                type_loss_weight = get_aux_task_weight(training_progress, gamma=10)
            else:
                type_loss_weight = self.params["type_loss_weight"]

            loss = (
                    self.params["category_loss_weight"] * answer_category_prediction_loss
                    + type_loss_weight * type_loss
            )

            type_probs = torch.sigmoid(type_prediction_scores)

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses/main",
                    tag_scalar_dict={
                        "answer_category_loss": answer_category_prediction_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )

                self.summary_writer.add_scalars(
                    main_tag="losses/types",
                    tag_scalar_dict={
                        "positive": type_loss_positives.item(),
                        "negative": type_loss_negatives.item(),
                    },
                    global_step=iteration_number,
                )

                average_positive_type_probability = safe_divide(
                    (type_probs * type_labels).sum().item(), type_labels.sum().item()
                )

                average_negative_type_probability = safe_divide(
                    (type_probs * (1 - type_labels)).sum().item(),
                    (1 - type_labels).sum().item(),
                )

                self.summary_writer.add_scalars(
                    main_tag="Average_type_probability",
                    tag_scalar_dict={
                        "positive_types": average_positive_type_probability,
                        "negative_types": average_negative_type_probability,
                    },
                    global_step=iteration_number,
                )

        all_scores = {"type_prediction_scores": type_prediction_scores, "answer_category_prediction_scores": answer_category_prediction_scores}

        return loss, all_scores

    def forward_inference(self, context_input):
        type_prediction_scores, answer_category_prediction_scores = self.score_candidate(
            text_vecs=context_input
        )

        return {"type_prediction_scores": type_prediction_scores, "answer_category_prediction_scores": answer_category_prediction_scores}


def get_type_model(model_number, params):
    if model_number == 1:
        # considers all types while computing type loss
        reranker = TypedBiEncoderRanker1(params)
        tokenizer = reranker.tokenizer
        model = reranker.model
    else:
        assert False, "Unsupported value given for type_model"

    return reranker, tokenizer, model
