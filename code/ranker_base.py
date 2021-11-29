# This file is adapted from https://github.com/facebookresearch/BLINK

from torch import nn
import torch


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class BertEncoder(nn.Module):
    def __init__(self, bert_model, output_dim, rep_for_ans_cat_pred, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.rep_for_ans_cat_pred = rep_for_ans_cat_pred
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            rep_for_type_prediction = self.additional_linear(self.dropout(embeddings))
        else:
            rep_for_type_prediction = embeddings

        # the "result" is used for dbpedia type prediction

        if self.rep_for_ans_cat_pred == "unused":
            # return the vector corresponding to unused1. By design, unused1 is the token after cls.
            rep_for_answer_category_prediction = output_bert[:, 1, :]
        elif self.rep_for_ans_cat_pred == "avg":
            # return the avg of the output word-piece-level embeddings (everything except the cls) for answer category prediction
            rep_for_answer_category_prediction = torch.mean(output_bert[:, 1:, :], dim=1)

        assert len(rep_for_answer_category_prediction.shape) == 2
        assert rep_for_answer_category_prediction.shape[0] == token_ids.shape[0]
        assert rep_for_answer_category_prediction.shape[1] == output_bert.shape[-1]

        return rep_for_type_prediction, rep_for_answer_category_prediction



