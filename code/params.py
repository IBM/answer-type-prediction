# This file is adapted from https://github.com/facebookresearch/BLINK

import argparse
import os


ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"


class BlinkParser(argparse.ArgumentParser):

    def __init__(
        self, add_blink_args=True, add_model_args=False, description="BLINK parser",
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler="resolve",
            formatter_class=argparse.HelpFormatter,
            add_help=add_blink_args,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ["BLINK_HOME"] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_blink_args:
            self.add_blink_args()
        if add_model_args:
            self.add_model_args()

    def add_blink_args(self, args=None):
        """
        Add common BLINK args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda",
            action="store_true",
            help="Whether not to use CUDA when available",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="random seed for initialization"
        )


    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")

        parser.add_argument(
            "--max_context_length",
            default=64,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )

        parser.add_argument(
            "--training_state_dir",
            default="training_state",
            type=str,
            required=False,
            help="Directory to save the training state, i.e, the model, optimizer, scheduler, etc",
        )
        parser.add_argument(
            "--training_state_file",
            default="training_state.pt",
            type=str,
            required=False,
            help="Name of the file having the training state (optimizer, scheduler, etc)",
        )

        # train_biencoder.py, biencoder.py, etc are using this. but dont use this from the command line if you want to resume training.
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--resume_training",
            action="store_true",
            help="Should the training be resumed from a previous checkpoint?",
        )

        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )

        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additional linear projection on top of BERT.",
        )

        parser.add_argument('--eval_set_paths', action='append', help='The paths to the test data', type=str)
        parser.add_argument(
            "--eval_only",
            action="store_true",
            help="Skip training, just evaluate on the data in eval_set_paths",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )

    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )

        parser.add_argument(
            "--tb",
            action="store_true",
            default=False,
            help="Enable tensorboard logging?",
        )

        parser.add_argument(
            "--processed_train_data_cache",
            default="train.cache",
            type=str,
            help="Preprocessed training data is stored here",
        )

        parser.add_argument(
            "--train_batch_size",
            default=8,
            type=int,
            help="Total batch size for training.",
        )
        parser.add_argument(
            "--eval_batch_size",
            default=8,
            type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=5, help="Interval of loss printing",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=999999999,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=200, help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", type=bool, default=False, help="Whether to shuffle train data",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")


    def add_type_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Arguments for type based model")
        parser.add_argument(
            "--type_model",
            default=1,
            type=int,
            help="Which type model should be used?",
        )

        parser.add_argument(
            "--num_types", default=60000, type=int, help="Length of type vocabulary",
        )

        parser.add_argument(
            "--type_embedding_dim",
            default=768,
            type=int,
            help="Length of type vocabulary",
        )
        parser.add_argument(
            "--freeze_type_embeddings",
            action="store_true",
            default=False,
            help="Passing this flag will not train the type embeddings",
        )
        parser.add_argument(
            "--type_embeddings_path",
            default="",
            type=str,
            help="Path to the bert, glove, etc embeddings of the types",
        )
        parser.add_argument(
            "--ontology_file",
            default="",
            type=str,
            help="Path to a file containing the list of child to parent edges",
        )
        parser.add_argument(
            "--no_linear_after_type_embeddings",
            action="store_true",
            default=False,
            help="Should there not be a linear layer on top of type embeddings?",
        )
        parser.add_argument("--category_loss_weight", default=1.0, type=float)
        parser.add_argument("--type_loss_weight", default=1.0, type=float)
        parser.add_argument("--type_loss_weight_positive", default=1.0, type=float)
        parser.add_argument("--type_loss_weight_negative", default=1.0, type=float)
        parser.add_argument(
            "--main_metric",
            type=str,
            choices=["cat_acc", "ndcg_5", "ndcg_10"],
            default="ndcg_10",
        )

        parser.add_argument(
            "--freeze_context_bert",
            action="store_true",
            default=False,
            help="Passing this flag will not train the context bert",
        )
        parser.add_argument(
            "--type_network_learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--type_task_importance_scheduling",
            type=str,
            choices=["none", "loss_weight", "grad_throttle"],
            default="none",
        )

        parser.add_argument(
            "--num_answer_categories",
            default=5,
            type=int,
            help="Number of answer categories",
        )
        parser.add_argument(
            "--type_id_to_name_file",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--smart_type_hierarchy_tsv",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--rep_for_ans_cat_pred",
            type=str,
            choices=["unused", "avg"],
            default="unused",
        )
        parser.add_argument(
            "--training_set_path",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--val_set_path",
            type=str,
            required=True,
        )


