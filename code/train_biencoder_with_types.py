# This file is adapted from https://github.com/facebookresearch/BLINK

import os
import torch
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule

from biencoder import get_type_model

import utils as utils
import data_process as data
from optimizer import get_type_model_optimizer
from params import BlinkParser

from validation_utils import evaluator_factory

from torch.utils.tensorboard import SummaryWriter

logger = None


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(42)

torch.autograd.set_detect_anomaly(True)


def save_training_state(
    training_state_file,
    output_dir,
    optimizer,
    scheduler,
    best_epoch_idx,
    best_score,
    steps_so_far,
    epoch_idx,
):
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_epoch_idx": best_epoch_idx,
        "best_score": best_score,
        "steps_so_far": steps_so_far,
        "start_epoch": epoch_idx,
    }
    training_state_save_file = os.path.join(output_dir, training_state_file)
    torch.save(training_state, training_state_save_file)


def load_training_state(training_state_file, output_dir):
    training_state = torch.load(os.path.join(output_dir, training_state_file))
    return training_state


def get_optimizer(model, params):
    return get_type_model_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
        additional_learning_rate=params["type_network_learning_rate"],
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    summary_writer = None
    if params["tb"]:
        summary_writer = SummaryWriter(log_dir=os.path.join(model_output_path, "tb"))

    # Init model
    reranker, tokenizer, model = get_type_model(params["type_model"], params)

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient
    # across `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    # seed = params["seed"]
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if reranker.n_gpu > 0:
    # torch.cuda.manual_seed_all(seed)

    if os.path.exists(
        os.path.join(params["output_path"], params["processed_train_data_cache"])
    ):
        logger.info("Loading processed training data from cache")
        train_data = None
        train_tensor_data = torch.load(
            os.path.join(params["output_path"], params["processed_train_data_cache"])
        )
    else:
        # Load train data and preprocess

        train_samples = utils.read_json(params["training_set_path"])
        logger.info("Read %d train samples." % len(train_samples))

        train_tensor_data = data.process_typed_data_factory(
            model_number=params["type_model"],
            params=params,
            samples=train_samples,
            tokenizer=tokenizer,
            logger=logger
        )
        torch.save(
            train_tensor_data,
            os.path.join(params["output_path"], params["processed_train_data_cache"]),
        )

    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    evaluator = evaluator_factory(
        model_number=params["type_model"],
        params=params,
        eval_dataset_path=params["val_set_path"],
        tokenizer=tokenizer,
        logger=logger,
        device=device
    )

    logger.info("-----------------------------------------")
    logger.info("Initial evaluation")
    # this line also logs the metrics to the screen
    results = evaluator.evaluate(model=reranker)
    logger.info("-----------------------------------------")

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1
    steps_so_far = 0
    start_epoch = 0

    num_train_epochs = params["num_train_epochs"]

    num_train_steps = (
        int(
            len(train_tensor_data)
            / params["train_batch_size"]
            / params["gradient_accumulation_steps"]
        )
        * num_train_epochs
    )

    stop_training = False

    if params["resume_training"]:
        # the model gets loaded in TypedBiEncoderRanker.__init__ when params["resume_training"] is True
        # the model is picked up from os.path.join(
        #         params["output_path"], params["training_state_dir"], "pytorch_model.bin"
        #     )

        assert (
            params["training_state_dir"] is not None
        ), "If training has to be resumed, both --resume_training and --training_state_dir must be specified"

        training_state = load_training_state(
            training_state_file=params["training_state_file"],
            output_dir=os.path.join(
                params["output_path"], params["training_state_dir"]
            ),
        )
        best_epoch_idx = training_state["best_epoch_idx"]
        best_score = training_state["best_score"]
        steps_so_far = training_state["steps_so_far"]
        start_epoch = training_state["start_epoch"]
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])

        logger.info(
            "Resuming training from epoch: {}, iteration: {}".format(
                start_epoch, steps_so_far
            )
        )

    if params["tb"]:
        reranker.set_summary_writer(summary_writer)

    for epoch_idx in trange(start_epoch, int(num_train_epochs), desc="Epoch"):

        if params["eval_only"]:
            logger.info("Eval only mode")
            stop_training = True

        if stop_training:
            break

        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):

            if (steps_so_far > num_train_steps) and (params["resume_training"]):
                logger.info(
                    "Reached maximum number of iterations ({}/{}). Stopping.".format(
                        steps_so_far, num_train_steps
                    )
                )
                stop_training = True
                break

            batch = tuple(t.to(device) for t in batch)

            if params["type_model"] in [1]:
                context_input, type_label_vec, answer_category_labels, is_type_resource = batch

                loss, _ = reranker(
                    context_input=context_input,
                    type_labels=type_label_vec,
                    answer_category_labels=answer_category_labels,
                    is_type_resource=is_type_resource,
                    iteration_number=steps_so_far,
                    training_progress=steps_so_far/num_train_steps
                )
            else:
                assert False, "Unsupported value of type_model"

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps_so_far += 1

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                # evaluate(
                #     reranker, valid_dataloader, params, device=device, logger=logger,
                # )
                results = evaluator.evaluate(model=reranker)
                model.train()
                logger.info("\n")

            if (step + 1) % (params["save_interval"] * grad_acc_steps) == 0:
                logger.info("Saving training state")
                logger.info("\n")
                training_state_output_folder_path = os.path.join(
                    model_output_path, params["training_state_dir"]
                )
                utils.save_model(model, tokenizer, training_state_output_folder_path)

                save_training_state(
                    training_state_file=params["training_state_file"],
                    output_dir=training_state_output_folder_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_epoch_idx=best_epoch_idx,
                    best_score=best_score,
                    steps_so_far=steps_so_far,
                    epoch_idx=epoch_idx,
                )

        results = evaluator.evaluate(model=reranker)

        if results["normalized_accuracy"] >= best_score:
            logger.info("***** Saving the new best model *****")
            utils.save_model(model, tokenizer, model_output_path)

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]

        if params["tb"]:
            summary_writer.add_scalar(
                tag="Val_accuracy",
                scalar_value=results["normalized_accuracy"],
                global_step=steps_so_far,
            )

        logger.info("***** Saving training state *****")

        training_state_output_folder_path = os.path.join(
            model_output_path, params["training_state_dir"]
        )
        utils.save_model(model, tokenizer, training_state_output_folder_path)

        save_training_state(
            training_state_file=params["training_state_file"],
            output_dir=training_state_output_folder_path,
            optimizer=optimizer,
            scheduler=scheduler,
            best_epoch_idx=best_epoch_idx,
            best_score=best_score,
            steps_so_far=steps_so_far,
            epoch_idx=epoch_idx
            + 1,  # +1 so that the training can resume from the "next" epoch
        )

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    logger.info("Best performance: {}".format(best_score))

    if params["tb"]:
        summary_writer.close()

    # if params["test_set_path"] != "":
    if params["eval_set_paths"] is not None:

        eval_set_metrics = {}

        # load the best model
        if os.path.isfile(os.path.join(model_output_path, "pytorch_model.bin")):
            params["path_to_model"] = os.path.join(model_output_path, "pytorch_model.bin")

        logger.info("Loading the mode from {}".format(params["path_to_model"]))
        reranker, tokenizer, model = get_type_model(params["type_model"], params)

        logger.info(
            "Evaluating on the test sets"
        )

        test_params = dict(params)

        for eval_dataset in params["eval_set_paths"]:

            logger.info("Evaluating on {}".format(eval_dataset))

            test_params = dict(params)
            # test_params["hard_negatives_file"] = None

            evaluator = evaluator_factory(
                model_number=params["type_model"],
                params=params,
                eval_dataset_path=eval_dataset,
                tokenizer=tokenizer,
                logger=logger,
                device=device
            )

            results = evaluator.evaluate(model=reranker, dump=True)

            eval_set_metrics[os.path.basename(eval_dataset)] = results

        # write eval_set_metrics to a file
        eval_set_metrics_file_path = os.path.join(test_params["output_path"], "eval_set_metrics.json")
        utils.write_json(eval_set_metrics, eval_set_metrics_file_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_type_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
