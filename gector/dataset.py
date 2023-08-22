from typing import List, Tuple
from collections import Counter
import torch
from tqdm import tqdm
import os
from transformers import PreTrainedTokenizer

class GECToRDataset:
    def __init__(
        self,
        srcs: List[str],
        d_labels: List[List[int]]=None,
        labels: List[List[int]]=None,
        word_masks: List[List[int]]=None,
        tokenizer: PreTrainedTokenizer=None,
        max_length:int=128
    ):
        self.tokenizer = tokenizer
        self.srcs = srcs
        self.d_labels = d_labels
        self.labels = labels
        self.word_masks = word_masks
        self.max_length = max_length
        self.label2id = None
        self.d_label2id = None
        
    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        src = self.srcs[idx]
        d_labels = self.d_labels[idx]
        labels = self.labels[idx]
        wmask = self.word_masks[idx]
        encode = self.tokenizer(
            src,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        return {
            'input_ids': encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze(),
            'd_labels': torch.tensor(d_labels).squeeze(),
            'labels': torch.tensor(labels).squeeze(),
            'word_masks': torch.tensor(wmask).squeeze()
        }

    def append_vocab(self, label2id, d_label2id):
        self.label2id = label2id
        self.d_label2id = d_label2id
        for i in range(len(self.labels)):
            self.labels[i] = [self.label2id.get(l, self.label2id['<OOV>']) for l in self.labels[i]]
            self.d_labels[i] = [self.d_label2id[l] for l in self.d_labels[i]]
    
    def get_labels_freq(self, exluded_labels: List[str] = []):
        assert(self.labels is not None and self.d_labels is not None)
        flatten_labels = [ll for l in self.labels for ll in l if ll not in exluded_labels]
        flatten_d_labels = [ll for l in self.d_labels for ll in l if ll not in exluded_labels]
        return Counter(flatten_labels), Counter(flatten_d_labels)

def align_labels_to_subwords(
    srcs: List[str],
    word_labels: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    batch_size: int=100000,
    max_length: int=128,
    keep_label: str='$KEEP',
    pad_token: str='<PAD>',
    correct_label: str='$CORRECT',
    incorrect_label: str='$INCORRECT'
):
    itr = list(range(0, len(srcs), batch_size))
    subword_labels = []
    subword_d_labels = []
    word_masks = []
    for i in tqdm(itr):
        encode = tokenizer(
            srcs[i:i+batch_size],
            max_length=max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        for i, wlabels in enumerate(word_labels[i:i+batch_size]):
            d_labels = []
            labels = []
            wmask = []
            word_ids = encode.word_ids(i)
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(pad_token)
                    d_labels.append(pad_token)
                    wmask.append(0)
                elif word_idx != previous_word_idx:
                    l = wlabels[word_idx]
                    labels.append(l)
                    wmask.append(1)
                    if l != keep_label:
                        d_labels.append(incorrect_label)
                    else:
                        d_labels.append(correct_label)
                else:
                    labels.append(pad_token)
                    d_labels.append(pad_token)
                    wmask.append(0)
                previous_word_idx = word_idx
            subword_d_labels.append(d_labels)
            subword_labels.append(labels)
            word_masks.append(wmask)
    return subword_d_labels, subword_labels, word_masks
        
def load_gector_format(
    input_file: str,
    delimeter: str='SEPL|||SEPR',
    additional_delimeter: str='SEPL__SEPR'
):  
    srcs = []
    word_level_labels = []  # the size will be (#sents, seq_length) if not get_interactive_tags,
                                # (#iteration, #sents, seq_length) if get_interactive_tags
    with open(input_file) as f:
        for line in f:
            src = [x.split(delimeter)[0] for x in line.split()]
            labels = [x.split(delimeter)[1] for x in line.split()]
            # Use only first tags. E.g. $REPLACE_meSEPL__SEPR$APPEND_too → $REPLACE_me
            labels = [l.split(additional_delimeter)[0] for l in labels]
            srcs.append(src)
            word_level_labels.append(labels)
    return srcs, word_level_labels

def load_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    delimeter: str='SEPL|||SEPR',
    additional_delimeter: str='SEPL__SEPR',
    batch_size: int=50000, # avoid too heavy computation in the tokenization
    max_length: int=128
):
    srcs, word_level_labels = load_gector_format(
        input_file,
        delimeter=delimeter,
        additional_delimeter=additional_delimeter
    )
    d_labels, labels, word_masks = align_labels_to_subwords(
        srcs,
        word_level_labels,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )
    return GECToRDataset(
        srcs=srcs,
        d_labels=d_labels,
        labels=labels,
        word_masks=word_masks,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    