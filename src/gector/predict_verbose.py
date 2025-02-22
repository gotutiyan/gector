import torch
import os
from tqdm import tqdm
from .modeling import GECToR
from transformers import PreTrainedTokenizer
from typing import List, Dict
from .predict import (
    edit_src_by_tags,
    _predict
)

def predict_verbose(
    model: GECToR,
    tokenizer: PreTrainedTokenizer,
    srcs: List[str],
    encode: dict,
    decode: dict,
    keep_confidence: float=0,
    min_error_prob: float=0,
    batch_size: int=128,
    n_iteration: int=5
) -> List[str]:
    srcs = [['$START'] + src.split(' ') for src in srcs]
    final_edited_sents = ['-1'] * len(srcs)
    to_be_processed = srcs
    original_sent_idx = list(range(0, len(srcs)))
    iteration_log: List[List[Dict]] = []  # [send_id][iteration_id]['src' or 'tags']
    iteration_log = []
    # Initialize iteration logs. 
    for i, src in enumerate(srcs):
        iteration_log.append([{
            'src': src,
            'tag': None
        }])
    for itr in range(n_iteration):
        print(f'Iteratoin {itr}. the number of to_be_processed: {len(to_be_processed)}')
        pred_labels, no_corrections = _predict(
            model,
            tokenizer,
            to_be_processed,
            keep_confidence,
            min_error_prob,
            batch_size
        )
        current_srcs = []
        current_pred_labels = []
        current_orig_idx = []
        for i, yes in enumerate(no_corrections):
            if yes: # there's no corrections?
                final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
            else:
                current_srcs.append(to_be_processed[i])
                current_pred_labels.append(pred_labels[i])
                current_orig_idx.append(original_sent_idx[i])
        if current_srcs == []:
            # Correcting for all sentences is completed.
            break
        edited_srcs = edit_src_by_tags(
            current_srcs,
            current_pred_labels,
            encode,
            decode
        )
        # Register the information during iteration.
        # edited_src will be the src of the next iteration.
        for i, orig_id in enumerate(current_orig_idx):
            iteration_log[orig_id][itr]['tag'] = current_pred_labels[i]
            iteration_log[orig_id].append({
                'src': edited_srcs[i],
                'tag': None
            })
        
        to_be_processed = edited_srcs
        original_sent_idx = current_orig_idx
        
        # print(f'=== Iteration {itr} ===')
        # print('\n'.join(final_edited_sents))
        # print(to_be_processed)
        # print(have_corrections)
    for i in range(len(to_be_processed)):
        final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
    assert('-1' not in final_edited_sents)
    return final_edited_sents, iteration_log