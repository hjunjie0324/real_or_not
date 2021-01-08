import numpy as np
import pandas as pd
import string
from collections import defaultdict
import re
import torch

from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_punc(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'',text)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)

def padding_to_maxLength(encodings,max_length):
    n = len(encodings['input_ids'])
    for index in range(n):
        length = len(encodings['input_ids'][index])
        for insert_index in range(length, max_length):
            encodings['input_ids'][index].insert(insert_index, 0)
            encodings['attention_mask'][index].insert(insert_index,0) 


def preprocess(filename):
    dataframe = pd.read_csv(filename)
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_emoji(x))
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_punc(x))
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_emoji(x))
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_html(x))
    
    text = dataframe['text'].tolist()
    target = dataframe['text'].tolist()

   
    #split data
    text_train, text_val, target_train, target_val = train_test_split(text, target, test_size = 0.1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(text_train, truncation = True, padding = True)
    max_length = 60
    padding_to_maxLength(train_encodings,max_length)

    val_encodings = tokenizer(text_val, truncation = True, padding = True)
    padding_to_maxLength(val_encodings, max_length)
        
    train_encodings.update({'target':target_train})
    val_encodings.update({'target':target_val})
    torch.save(val_encodings,'val_encodings.pt')
    return train_encodings

def preprocess_for_test(filename):
    dataframe = pd.read_csv(filename)
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_emoji(x))
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_punc(x))
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_emoji(x))
    dataframe['text'] = dataframe['text'].apply(lambda x:remove_html(x))
    
    text = dataframe['text'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(text, truncation = True, padding = True)
    max_length = 60
    padding_to_maxLength(encodings,max_length)

    return encodings

if __name__ == '__main__':
    train_df = pd.read_csv("train.csv")
    encodings = preprocess(train_df)
    print(encodings['input_ids'][0])

    