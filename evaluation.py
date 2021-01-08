import predict
import torch
from train import RealOrNotDataset

def evaluation(prediction, label):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    
    return 

if __name__ == "__main__":
    val_encodings = torch.load('val_encodings.pt')
    val_dataset = RealOrNotDataset(val_encodings)
    prediction = predict.predict(val_dataset)
    label = val_encodings['target']
    evaluation(prediction, label)
