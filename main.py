'''
1. Load the dataset
2. Preprocess the dataset
3. Create NN module
4. Define optimizer, loss function, eval
5. Train
'''
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

from preprocess import *
from constants import *
from net import Transformer
from torch.optim import Adam
from train import *

device = "cuda" if torch.cuda.is_available else "cpu"

def trainer():
    model = Transformer(num_encoder=NUM_ENCODER, num_decoder=NUM_DECODER, d_model=D_MODEL, num_heads=NUM_HEADS, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    train_epoch(model, optimizer)
    
if __name__ == "__main__":
    trainer()

