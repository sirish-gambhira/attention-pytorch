from preprocess import *
from net import Transformer
from torch.optim import Adam
from torch.utils.data import DataLoader

def train_epoch(model : Transformer, optimizer: Adam):
    train_data = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    for src, tgt in train_dataloader:
        model(src, tgt)
        break
        