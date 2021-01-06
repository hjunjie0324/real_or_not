import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier,self).__init__()
        D_in = 768
        D_hidden = 50
        D_out = 2
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_hidden),
            nn.ReLU(),
            nn.Linear(D_hidden, D_out)
        )

    def forward(self,input_ids,attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = bert_output[0][:,0,:]
        output = self.classifier(last_hidden_state_cls)

        return output

