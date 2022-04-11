
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric
from functools import partial
from predict_function import predict, predict_and_early_stopping
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from textbrewer.distiller_utils import move_to_device

class Distillation:

    
    def __init__(self, task_name, base_model_name, output_dir_prefix):
        self.task_name = task_name
        self.base_model_name = base_model_name
        self.output_dir_prefix = output_dir_prefix
        
        

    def load_tokenizer_dataset_preprocess(self):
        self.train_dataset = load_dataset('glue', self.task_name, split='train')
        self.val_dataset = load_dataset('glue', self.task_name, split='validation_matched')
        if self.task_name == "mnli":
            self.val_mm_dataset = load_dataset('glue', self.task_name, split='validation_mismatched')
        self.test_dataset = load_dataset('glue', self.task_name, split='test_matched')



        self.train_dataset = self.train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.val_dataset = self.val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        if self.task_name == "mnli":
            self.val_mm_dataset = self.val_mm_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        self.test_dataset = self.test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

        self.val_dataset = self.val_dataset.remove_columns(['label'])
        if self.task_name == "mnli":
            self.val_mm_dataset = self.val_mm_dataset.remove_columns(['label'])
        self.test_dataset = self.test_dataset.remove_columns(['label'])
        self.train_dataset = self.train_dataset.remove_columns(['label'])


        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)


        task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
        }
        sentence_keys = task_to_keys[self.task_name]
        MAX_LENGTH = 128
        self.train_dataset = self.train_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        self.val_dataset = self.val_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        if self.task_name == "mnli":
            self.val_mm_dataset = self.val_mm_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
        self.test_dataset = self.test_dataset.map(lambda e: self.tokenizer(e[sentence_keys[0]],e[sentence_keys[1]], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

        if "roberta" in self.base_model_name.lower():
            input_columns = ['input_ids', 'attention_mask', 'labels']
        else:   
            input_columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
            
        self.train_dataset.set_format(type='torch', columns=input_columns)
        self.val_dataset.set_format(type='torch', columns=input_columns)
        if self.task_name == "mnli":
            self.val_mm_dataset.set_format(type='torch', columns=input_columns)
        self.test_dataset.set_format(type='torch', columns=input_columns)


    def distill(self, teacher_model_path, num_epochs, num_hidden_layers, hidden_size = 768,temperature = 4, batch_size= 128, intermediate_matches=None, evaluate_teacher=True):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        #device = 'cpu'

        # config
        teacher_config = AutoConfig.from_pretrained(self.base_model_name)
        teacher_config.output_hidden_states = True
        if self.task_name == "mnli":
            teacher_config.num_labels = 3
        elif self.task_name == "stsb":
            teacher_config.num_labels = 1
        
        if teacher_model_path.endswith(".pt"):
            teacher_model = AutoModelForSequenceClassification.from_config(teacher_config)
            teacher_model.load_state_dict(torch.load(teacher_model_path))
        else:
            teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_path, output_hidden_states = True)


        teacher_model = teacher_model.to(device=device)



        student_config = AutoConfig.from_pretrained(self.base_model_name)
        student_config.output_hidden_states = True
        if self.task_name == "mnli":
            student_config.num_labels = 3
        elif self.task_name == "stsb":
            student_config.num_labels = 3

        student_config.num_hidden_layers = num_hidden_layers
        #student_config.hidden_size = hidden_size
        #student_config.num_attention_heads = hidden_size / 64
        #student_config.intermediate_size = hidden_size * 4
        
        self.student_config = student_config

        continue_training = False
        student_model = AutoModelForSequenceClassification.from_config(student_config)
        if continue_training:
            student_model.load_state_dict(torch.load(''))
        student_model = student_model.to(device=device)
        
        
        assert teacher_model.config.vocab_size == student_model.config.vocab_size
        assert teacher_model.config.vocab_size == len(self.tokenizer)

        if evaluate_teacher:
            eval_dataloader = DataLoader(self.val_dataset, batch_size=32)

            metric= load_metric("glue",self.task_name)
            teacher_model.to(device)
            teacher_model.eval()
            for batch in eval_dataloader:
                batch = {k: v for k, v in batch.items()}
                batch = move_to_device(batch,device)
                with torch.no_grad():
                    outputs = teacher_model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
            print("Teacher model validation dataset score:")
            print(metric.compute())



        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128) #prepare dataloader
        num_training_steps = len(train_dataloader) * num_epochs
        # Optimizer and learning rate scheduler
        optimizer = AdamW(student_model.parameters(), lr=1e-4)

        scheduler_class = get_linear_schedule_with_warmup
        # arguments dict except 'optimizer'
        scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


        def simple_adaptor(batch, model_outputs):
            return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}

        if intermediate_matches:
            from matches import matches
            match_list_L4t = ["L4t_hidden_mse", "L4_hidden_smmd"]
            match_list_L3 = ["L3_hidden_mse", "L3_hidden_smmd"]
            intermediate_matches = []
            for match in match_list_L3:
                intermediate_matches += matches[match]

        output_dir = self.output_dir_prefix + self.base_model_name + "/" + self.task_name + "/" + "hl"+ str(student_model.config.num_hidden_layers) + "_hs" +  str(student_model.config.hidden_size) + "_tp" + str(temperature) + "_bs" + str(batch_size) +  "/"
        distill_config = DistillationConfig(
            intermediate_matches=intermediate_matches
        )
        train_config = TrainingConfig(device=device, output_dir = output_dir + "models/")



        # prepare callback function
        local_rank = -1
        predict_batch_size = 32
        device = device
        do_train_eval = True
        
        if self.task_name == "mnli":
            eval_datasets = [self.val_dataset,self.val_mm_dataset]
        else:
            eval_datasets = [self.val_dataset]

        callback_func = partial(predict_and_early_stopping, eval_datasets=eval_datasets, output_dir=output_dir+"results/",
                                task_name=self.task_name, local_rank=local_rank,
                                predict_batch_size=predict_batch_size,
                                device=device, do_train_eval=do_train_eval, train_dataset=self.train_dataset.select(range(10000)))

        distiller = GeneralDistiller(
            train_config=train_config, distill_config=distill_config,
            model_T=teacher_model, model_S=student_model, 
            adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


        with distiller:
            distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=callback_func)
        self.student_model = student_model

        def test_student(self, test_model_path=None):

            if test_model_path:
                test_model = AutoModelForSequenceClassification.from_config(self.student_config)
                test_model.load_state_dict( torch.load(test_model_path) )
            else:
                test_model = self.student_model

            eval_dataloader = DataLoader(self.val_dataset, batch_size=32)



            metric= load_metric("glue",self.task_name)
            test_model.eval()
            for batch in eval_dataloader:
                batch = {k: v for k, v in batch.items()}
                with torch.no_grad():
                    outputs = test_model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            return metric.compute()




