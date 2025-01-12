from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from constants import *
import torch

# Source: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/c64c91cf87c13c0e83586b8e66e4d74e/translation_transformer.ipynb#scrollTo=XOOnxA9eoift

token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:     # type: ignore
	for data_sample in data_iter:
		yield token_transform[language](data_sample[language_index[language]])

def get_vocab_transform() -> dict:
	vocab_transform = {}
	for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
		train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
		vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
														min_freq=1,
														specials=special_symbols,
														special_first=True)
	for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
		vocab_transform[ln].set_default_index(UNK_IDX)
	
	return vocab_transform

vocab_transform = get_vocab_transform()
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

# helper function to club together sequential operations
def sequential_transforms(*transforms):
	def func(txt_input):
		for transform in transforms:
			txt_input = transform(txt_input)
		return txt_input
	return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]) -> torch.Tensor:
	return torch.cat((torch.tensor([BOS_IDX]),
					  torch.tensor(token_ids),
					  torch.tensor([EOS_IDX])))

def get_text_transform() -> dict:
	text_transform = {}
	for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
		text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
											   vocab_transform[ln], #Numericalization
											   tensor_transform) # Add BOS/EOS and create tensor
	return text_transform

text_transform = get_text_transform()

# function to collate data samples into batch tensors
def collate_fn(batch):
	src_batch, tgt_batch = [], []
	for src_sample, tgt_sample in batch:
		src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
		tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
  
	src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
	tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
	return src_batch, tgt_batch
