import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
import config as Conf
BertLayerNorm = torch.nn.LayerNorm
from transformers import BertModel, RobertaModel
logger = logging.getLogger(__name__)


def initializer_builder(std):
    _std = std
    def init_bert_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_bert_weights

class BertForGLUESimple(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForGLUESimple, self).__init__()

        config.num_labels = num_labels
        self.num_labels = num_labels
        config.output_hidden_states = True
        config.output_attentions = False
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        initializer = initializer_builder(config.initializer_range)
        self.apply(initializer)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        #last_hidden_state, pooled_output, hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        
        output_for_cls = self.dropout(pooled_output)
        logits  = self.classifier(output_for_cls)  # output size: batch_size,num_labels
        #assert len(sequence_output)==self.bert.config.num_hidden_layers + 1  # embeddings + 12 hiddens
        #assert len(attention_output)==self.bert.config.num_hidden_layers + 1 # None + 12 attentions
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits,labels)
            return logits, hidden_states, loss
        else:
            return logits



def BertForGLUESimpleAdaptor(batch, model_outputs, with_logits=True, with_mask=False):
    print("##### Model outputs")
    for i,e in enumerate(model_outputs):
        if i ==2:
            print(e)
            print("Output ", i, " size: ", e.size())
            continue
        for j,t in enumerate(e):
            if isinstance(t, torch.Tensor):
                print("Output ", i, " Tupel " ,j ," size: ", t.size())
            else:
                print("Output ", i, " Tupel " ,j ," type: ", type(t))
            
    print("")
        
    dict_obj = {'hidden': model_outputs[1]}
    #dict_obj = {'hidden': model_outputs.hidden_states}
    if with_mask:
        dict_obj['inputs_mask'] = batch[1]
    if with_logits:
        dict_obj['logits'] = (model_outputs[0],)
        #dict_obj['logits'] = (model_outputs.logits,)
    return dict_obj

def BertForGLUESimpleAdaptorTrain(batch, model_outputs):
    return {'losses':(model_outputs[2],)}
    #return {'losses':(model_outputs.loss,)}
