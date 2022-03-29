import numpy as np
import os
import torch
from torch.utils.data import SequentialSampler,DistributedSampler,DataLoader
from utils_glue import compute_metrics
from tqdm import tqdm
import logging
from collections import defaultdict
logger = logging.getLogger(__name__)
import datasets
from transformers import PretrainedConfig
from textbrewer.distiller_utils import move_to_device


def predict(model,eval_datasets,step,output_dir,task_name,local_rank,predict_batch_size,device, do_train_eval=False, train_dataset=None):
    eval_task_names = [task_name]
    eval_task_names += "mnli-mm" if task_name == "mnli"
    eval_task_names += "train" if do_train_eval
    eval_datasets += train_dataset if do_train_eval
    eval_output_dir = output_dir
    task_results = {}
    for eval_task,eval_dataset in zip(eval_task_names, eval_datasets):
        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=predict_batch_size)
        model.eval()
        """
        if not (task_name == "stsb" or task_name == "sts-b"):
            pred_logits = []
            label_ids = []
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
                #input_ids=batch["input_ids"].to(device)
                #input_mask = batch["attention_mask"].to(device)
                #segment_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"]
                batch = move_to_device(batch, device)
                with torch.no_grad():
                    #outputs = model(input_ids, input_mask)#, segment_ids)
                    outputs = model(**batch)
                    logits = outputs.logits
                pred_logits.append(logits.detach().cpu())
                label_ids.append(labels)
            pred_logits = np.array(torch.cat(pred_logits),dtype=np.float32)
            label_ids = np.array(torch.cat(label_ids),dtype=np.int64)

            preds = np.argmax(pred_logits, axis=1)
            results = compute_metrics(eval_task, preds, label_ids)

            logger.info("***** Eval results {} task {} *****".format(step, eval_task))
            for key in sorted(results.keys()):
                logger.info(f"{eval_task} {key} = {results[key]:.5f}")
            task_results[eval_task] = results
        else:
        """
        metric = datasets.load_metric("glue",task_name)
        for batch in eval_dataloader:
            labels = batch["labels"]
            batch = move_to_device(batch, device)
            model_predictions = model(**batch)["logits"]
            if task_name !="stsb":
                model_predictions = np.argmax(model_predictions, axis=1)
            metric.add_batch(predictions=model_predictions, references=labels)
        results = metric.compute()
        logger.info("***** Eval results {} task {} *****".format(step, eval_task))
        for key in sorted(results.keys()):
            logger.info(f"{eval_task} {key} = {results[key]:.5f}")

        task_results[eval_task] = results
    
    
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    write_results(output_eval_file,step,task_results,eval_task_names)
    model.train()
    return task_results

def write_results_stsb(output_eval_file,step,task_results,eval_task_names):
    with open(output_eval_file, "a") as writer:
        writer.write(f"step: {step:<8d} ")
        line = "Corr.:"
        for eval_task in eval_task_names:
            for key in task_results[eval_task].keys():
                res = task_results[eval_task][key]
                line += f"{key}={res:.5f} "
        line += "\n"
        writer.write(line)
                
def write_metric_results(output_eval_file,step, metric_score,eval_task_names):
    with open(output_eval_file, "a") as writer:
        writer.write(f"step: {step:<8d} ")
        line = ""
        for eval_task in eval_task_names:
            line+= eval_task + " " + str(metric_score)
        line += "\n"
        writer.write(line)
    

def write_results(output_eval_file,step,task_results,eval_task_names):
    with open(output_eval_file, "a") as writer:
            all_acc = 0
            writer.write(f"step: {step:<8d} ")
            line = "Acc:"

            for eval_task in eval_task_names:
                acc = task_results[eval_task]['acc']
                all_acc += acc
                line += f"{eval_task}={acc:.5f} "
            all_acc /= len(eval_task_names)
            line += f"All={all_acc:.5f}\n"
            writer.write(line)

def predict_ens(models,eval_datasets,step,args):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_output_dir = args.output_dir
    task_results = {}
    for eval_task,eval_dataset in zip(eval_task_names, eval_datasets):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)
        for model in models:
            model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            input_ids, input_mask, segment_ids, labels = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)

            with torch.no_grad():
                logits_list = [model(input_ids, input_mask, segment_ids) for model in models]
            logits = sum(logits_list)/len(logits_list)
            pred_logits.append(logits.detach().cpu())
            label_ids.append(labels)
        pred_logits = np.array(torch.cat(pred_logits),dtype=np.float32)
        label_ids = np.array(torch.cat(label_ids),dtype=np.int64)

        preds = np.argmax(pred_logits, axis=1)
        results = compute_metrics(eval_task, preds, label_ids)

        logger.info("***** Eval results {} task {} *****".format(step, eval_task))
        for key in sorted(results.keys()):
            logger.info(f"{eval_task} {key} = {results[key]:.5f}")
        task_results[eval_task] = results

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    write_results(output_eval_file,step,task_results,eval_task_names)
    for model in models:
        model.train()
    return task_results
