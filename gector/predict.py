import torch
import os
from tqdm import tqdm
from .modeling import GECToR
from transformers import PreTrainedTokenizer
from typing import List

def load_verb_dict(verb_file: str):
    path_to_dict = os.path.join(verb_file)
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = word2
    return encode, decode

def edit_src_by_tags(
    srcs: List[List[str]],
    pred_labels: List[List[str]],
    encode: dict,
    decode: dict
) -> List[str]:
    edited_srcs = []
    for tokens, labels in zip(srcs, pred_labels):
        edited_tokens = []
        for t, l, in zip(tokens, labels):
            n_token = process_token(t, l, encode, decode)
            if n_token == None:
                n_token = t
            edited_tokens += n_token.split(' ')
        if len(tokens) > len(labels):
            omitted_tokens = tokens[len(labels):]
            edited_tokens += omitted_tokens
        temp_str = ' '.join(edited_tokens) \
            .replace(' $MERGE_HYPHEN ', '-') \
            .replace(' $MERGE_SPACE ', '') \
            .replace(' $DELETE', '') \
            .replace('$DELETE ', '')
        edited_srcs.append(temp_str.split(' '))
    return edited_srcs

def process_token(
    token: str,
    label: str,
    encode: dict,
    decode: dict
) -> str:
    if '$APPEND_' in label:
        return token + ' ' + label.replace('$APPEND_', '')
    elif token == '$START':
        return token
    elif label in ['<PAD>', '<OOV>', '$KEEP']:
        return token
    elif '$TRANSFORM_' in label:
        return g_transform_processer(token, label, encode, decode)
    elif '$REPLACE_' in label:
        return label.replace('$REPLACE_', '')
    elif label == '$DELETE':
        return label
    elif '$MERGE_' in label:
        return token + ' ' + label
    else:
        return token
    
def g_transform_processer(
    token: str,
    label: str,
    encode: dict,
    decode: dict
) -> str:
    # Case related
    if label == '$TRANSFORM_CASE_LOWER':
        return token.lower()
    elif label == '$TRANSFORM_CASE_UPPER':
        return token.upper()
    elif label == '$TRANSFORM_CASE_CAPITAL':
        return token.capitalize()
    elif label == '$TRANSFORM_CASE_CAPITAL_1':
        if len(token) <= 1:
            return token
        return token[0] + token[1:].capitalize()
    elif label == '$TRANSFORM_AGREEMENT_PLURAL':
        return token + 's'
    elif label == '$TRANSFORM_AGREEMENT_SINGULAR':
        return token[:-1]
    elif label == '$TRANSFORM_SPLIT_HYPHEN':
        return ' '.join(token.split('-'))
    else:
        encoding_part = f"{token}_{label[len('$TRANSFORM_VERB_'):]}"
        decoded_target_word = decode.get(encoding_part)
        return decoded_target_word

def get_word_masks_from_word_ids(
    word_ids: List[List[int]],
    n: int
):
    word_masks = []
    for i in range(n):
        previous_id = 0
        mask = []
        for _id in word_ids(i):
            if _id is None:
                mask.append(0)
            elif previous_id != _id:
                mask.append(1)
            else:
                mask.append(0)
            previous_id = _id
        word_masks.append(mask)
    return word_masks

def _predict(
    model: GECToR,
    tokenizer: PreTrainedTokenizer,
    srcs: List[str],
    keep_confidence: float=0,
    min_error_prob: float=0,
    batch_size: int=128
):
    itr = list(range(0, len(srcs), batch_size))
    pred_labels = []
    no_corrections = []
    no_correction_ids = [model.config.label2id[l] for l in ['$KEEP', '<OOV>', '<PAD>']]
    for i in tqdm(itr):
        # The official models was trained without special tokens, e.g. [CLS] [SEP].
        batch = tokenizer(
            srcs[i:i+batch_size],
            return_tensors='pt',
            max_length=model.config.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=not model.config.is_official_model
        )
        batch['word_masks'] = torch.tensor(
            get_word_masks_from_word_ids(
                batch.word_ids,
                batch['input_ids'].size(0)
            )
        )
        word_ids = batch.word_ids
        if torch.cuda.is_available():
            batch = {k:v.cuda() for k,v in batch.items()}
        outputs = model.predict(
            batch['input_ids'],
            batch['attention_mask'],
            batch['word_masks'],
            keep_confidence,
            min_error_prob
        )
        # Align subword-level label to word-level label
        for i in range(len(outputs.pred_labels)):
            no_correct = True
            labels = []
            previous_word_idx = None
            for j, idx in enumerate(word_ids(i)):
                if idx is None:
                    continue
                if idx != previous_word_idx:
                    labels.append(outputs.pred_labels[i][j])
                    if outputs.pred_label_ids[i][j] not in no_correction_ids:
                        no_correct = False
                previous_word_idx = idx
            # print(no_correct, labels)
            pred_labels.append(labels)
            no_corrections.append(no_correct)
    # print(pred_labels)
    return pred_labels, no_corrections

def predict(
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
        # if itr > 2:
        #     for l in current_pred_labels:
        #         print(l)
        edited_srcs = edit_src_by_tags(
            current_srcs,
            current_pred_labels,
            encode,
            decode
        )
        to_be_processed = edited_srcs
        original_sent_idx = current_orig_idx
        
        # print(f'=== Iteration {itr} ===')
        # print('\n'.join(final_edited_sents))
        # print(to_be_processed)
        # print(have_corrections)
    for i in range(len(to_be_processed)):
        final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
    assert('-1' not in final_edited_sents)
    return final_edited_sents