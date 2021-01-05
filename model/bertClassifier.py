import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Linear(768,2)

    def forward(self,input_ids,attention_mask,target):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = bert_output[0][:,0,:]
        print("bert output shape:",bert_output[0].shape)
        output = self.classifier(last_hidden_state_cls)

        return output

