#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/lokwq/TextBrewer/blob/add_note_examples/sst2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# This notebook shows how to fine-tune a model on sst-2 dataset and how to distill the model with TextBrewer.
# 
# Detailed Docs can be find here:
# https://github.com/airaria/TextBrewer

# In[1]:


import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


# In[2]:


import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig, AutoModelForSequenceClassification,RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric
from functools import partial
from predict_function import predict


# ### Prepare dataset to train

# In[34]:


train_dataset = load_dataset('glue', 'mnli', split='train')#,cache_dir="/work/mhessent/cache")
val_dataset = load_dataset('glue', 'mnli', split='validation_matched')#,cache_dir="/work/mhessent/cache")
val_mm_dataset = load_dataset('glue', 'mnli', split='validation_mismatched')
test_dataset = load_dataset('glue', 'mnli', split='test_matched')#,cache_dir="/work/mhessent/cache")


# In[35]:


train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_mm_dataset = val_mm_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

val_dataset = val_dataset.remove_columns(['label'])
val_mm_dataset = val_mm_dataset.remove_columns(['label'])
test_dataset = test_dataset.remove_columns(['label'])
train_dataset = train_dataset.remove_columns(['label'])


# In[36]:


#model = BertForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
tokenizer = RobertaTokenizer.from_pretrained("/work/mhessent/master_thesis/eval_out/roberta-base/mnli/lr3e-05_bs32_epochs10/checkpoint-49088")


# In[37]:


MAX_LENGTH = 128
train_dataset = train_dataset.map(lambda e: tokenizer(e['premise'],e['hypothesis'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_dataset = val_dataset.map(lambda e: tokenizer(e['premise'],e['hypothesis'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_mm_dataset = val_mm_dataset.map(lambda e: tokenizer(e['premise'],e['hypothesis'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['premise'],e['hypothesis'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)


# In[38]:


train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_mm_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[39]:


train_dataset[:]["labels"].unique()


# In[40]:



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    metric = load_metric("glue","mnli")
    return metric.compute(predictions=preds, references=labels)


# In[41]:


#start training 
"""
training_args = TrainingArguments(
    output_dir='outputs/results',          #output directory
    learning_rate=13e-5,
    num_train_epochs=3,              
    per_device_train_batch_size=32,                #batch size per device during training
    per_device_eval_batch_size=32,                #batch size for evaluation
    logging_dir='outputs/logs',            
    logging_steps=100,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=True,
    # eval_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics
)

train_out = trainer.train()
"""
#after training, you could find traing logs and checpoints in your own dirve. also you can reset the file address in training args


# In[42]:


#torch.save(model.state_dict(), 'outputs/mnli_teacher_model.pt')


# ### Start distillation

# In[43]:


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128) #prepare dataloader


# In[44]:


import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup


# In[ ]:





# import textbrewer
# from textbrewer import GeneralDistiller
# from textbrewer import TrainingConfig, DistillationConfig
# from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer, RobertaConfig, RobertaForSequenceClassification
# from transformers import get_linear_schedule_with_warmupInitialize the student model by BertConfig and prepare the teacher model.
# 
# bert_config_L3.json refers to a 3-layer Bert.
# 
# bert_config.json refers to a standard 12-layer Bert.

# In[45]:


#hub_model = RobertaForSequenceClassification.from_pretrained("/work/mhessent/master_thesis/eval_out/roberta-base/mnli/lr2e-05_bs32_epochs10/checkpoint-49088")
#torch.save(hub_model.state_dict(), 'outputs/hub_roberta_mnli_teacher_model.pt')


# In[46]:


config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
#bert_config = BertConfig.from_json_file('/work/mhessent/TextBrewer/examples/student_config/bert_base_cased_config/bert_config.json')
config.output_hidden_states = True
#bert_config.vocab_size = 30522
config.num_labels = 3
teacher_model = RobertaForSequenceClassification(config) #, num_labels = 2
#teacher_model.load_state_dict(torch.load('outputs/mnli_teacher_model.pt'))
teacher_model.load_state_dict(torch.load('/work/mhessent/master_thesis/eval_out/roberta-base/mnli/lr2e-05_bs32_epochs4/torch_state_dict.pt'))
                             
"""
model = BertForSequenceClassification.from_pretrained("/work/mhessent/master_thesis/eval_out/bert-base-uncased/mnli/lr3e-05_bs32_epochs3/checkpoint-36816")
torch.save(model.state_dict(), 'outputs/hub_mnli_teacher_model.pt')
bert_config = BertConfig.from_json_file('/work/mhessent/TextBrewer/examples/student_config/bert_base_cased_config/bert_config.json')
bert_config.output_hidden_states = True
bert_config.vocab_size = 30522
bert_config.num_labels = 3
teacher_model = BertForSequenceClassification(bert_config) #, num_labels = 2
teacher_model.load_state_dict(torch.load('outputs/hub_mnli_teacher_model.pt'))
"""


teacher_model = teacher_model.to(device=device)



student_config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
student_config.output_hidden_states = True
student_config.num_labels = 3
student_config.num_hidden_layers = 3
#student_config.vocab_size = teacher_model.config.vocab_size

continue_training = False
student_model = RobertaForSequenceClassification(student_config)
if continue_training:
    student_model.load_state_dict(torch.load('/work/mhessent/TextBrewer/examples/notebook_examples/saved_models/gs490880.pkl'))
student_model = student_model.to(device=device)


print(teacher_model.config.vocab_size)
print(student_model.config.vocab_size)
print(len(tokenizer))


# In[47]:



from torch.utils.data import DataLoader
eval_dataloader = DataLoader(val_dataset, batch_size=8)
from textbrewer.distiller_utils import move_to_device

metric= load_metric("glue","mnli")
#teacher_model.cpu()
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

print(metric.compute())



# The cell below is to distill the teacher model to student model you prepared.
# 
# After the code execution is complete, the distilled model will be in 'saved_model' in colab file list

# In[49]:


num_epochs = 60
num_training_steps = len(train_dataloader) * num_epochs
# Optimizer and learning rate scheduler
optimizer = AdamW(student_model.parameters(), lr=1e-4)

scheduler_class = get_linear_schedule_with_warmup
# arguments dict except 'optimizer'
scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}


from matches import matches
intermediate_matches = None
match_list_L4t = ["L4t_hidden_mse", "L4_hidden_smmd"]
match_list_L3 = ["L3_hidden_mse", "L3_hidden_smmd"]
intermediate_matches = []
for match in match_list_L3:
    intermediate_matches += matches[match]

distill_config = DistillationConfig()
    #intermediate_matches=intermediate_matches)
train_config = TrainingConfig(device=device)



task_name = "mnli"
local_rank = -1
predict_batch_size = 32
device = device
output_dir = "outputs/" + task_name + "/" 
eval_datasets = [val_dataset,val_mm_dataset]

callback_func = partial(predict, eval_datasets=eval_datasets, output_dir=output_dir,task_name=task_name,local_rank=local_rank,predict_batch_size=predict_batch_size,device=device)

distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model, 
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


with distiller:
    distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=callback_func)


# In[ ]:


test_model = BertForSequenceClassification(bert_config_T3)
test_model.load_state_dict(torch.load('/work/mhessent/TextBrewer/examples/notebook_examples/saved_models/gs490880.pkl'))#gs4210 is the distilled model weights file


# In[ ]:


from torch.utils.data import DataLoader
eval_dataloader = DataLoader(val_dataset, batch_size=8)


# In[ ]:


from textbrewer.distiller_utils import move_to_device


# In[ ]:


metric= load_metric("accuracy")
test_model.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()


# In[ ]:




