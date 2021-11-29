import torch

pre_trained_model_path = "./model/pytorch_model.bin"
bert_embeddings_path = "./791_types.t7"
out_embeddings_path = "./768_out_of_791_types_pretrained.t7"


pre_trained_model = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))

bert_embeddings = torch.load(bert_embeddings_path, map_location=torch.device('cpu'))

new_embeddings = torch.cat([pre_trained_model["additional_type_embeddings.weight"], bert_embeddings[768 : ]], dim =0)
print(new_embeddings.shape)

torch.save(new_embeddings, out_embeddings_path)