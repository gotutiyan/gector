import torch
import os
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        for t, l in zip(tokens, labels):
            n_token = process_token(t, l, encode, decode)
            if n_token is None:
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
    keep_confidence: float = 0,
    min_error_prob: float = 0,
    batch_size: int = 128
) -> Tuple[List[List[str]], List[bool], List[List[float]]]:
    """
    Predict edit tags and confidence scores for a batch of sentences.
    
    Returns:
        Tuple of (predicted labels, no-correction flags, per-token confidence scores)
    """
    if not srcs:
        logger.warning("Empty input list provided to _predict")
        return [], [], []
    
    itr = list(range(0, len(srcs), batch_size))
    pred_labels = []
    no_corrections = []
    confidences = []
    no_correction_ids = [model.config.label2id[l] for l in ['$KEEP', '<OOV>', '<PAD>']]
    
    for i in itr:
        batch = tokenizer(
            srcs[i:i + batch_size],
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
            batch = {k: v.cuda() for k, v in batch.items()}
        
        # Get model predictions
        outputs = model.predict(
            batch['input_ids'],
            batch['attention_mask'],
            batch['word_masks'],
            keep_confidence,
            min_error_prob
        )
        
        # Debug: Log attributes of outputs
        logger.debug(f"GECToRPredictionOutput attributes: {dir(outputs)}")
        
        # Try to get logits
        try:
            with torch.no_grad():
                model_outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    word_masks=batch['word_masks']
                )
                if hasattr(model_outputs, 'logits'):
                    pred_probs = torch.softmax(model_outputs.logits, dim=-1)
                    logger.debug("Logits retrieved successfully from model forward pass")
                else:
                    logger.warning("No logits in model output. Falling back to heuristic.")
                    pred_probs = None
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning(f"Could not retrieve logits: {e}. Using heuristic confidence scores.")
            pred_probs = None
        
        # Align subword-level labels and confidences to word-level
        for i in range(len(outputs.pred_labels)):
            no_correct = True
            labels = []
            conf_scores = []
            previous_word_idx = None
            edit_count = 0
            for j, idx in enumerate(word_ids(i)):
                if idx is None:
                    continue
                if idx != previous_word_idx:
                    label = outputs.pred_labels[i][j]
                    labels.append(label)
                    if pred_probs is not None:
                        conf_score = pred_probs[i][j][outputs.pred_label_ids[i][j]].item()
                    else:
                        # Heuristic: Lower scores for edits to avoid constant 1.0
                        if label in ['$KEEP', '<PAD>', '<OOV>']:
                            conf_score = 0.95  # Slightly less than 1.0 for no edits
                        elif label in ['$DELETE', '$MERGE_HYPHEN', '$MERGE_SPACE']:
                            conf_score = 0.7   # Major edits
                        else:
                            conf_score = 0.85  # Minor edits (APPEND, TRANSFORM, REPLACE)
                        edit_count += 1 if label not in ['$KEEP', '<PAD>', '<OOV>'] else 0
                    conf_scores.append(conf_score)
                    if outputs.pred_label_ids[i][j] not in no_correction_ids:
                        no_correct = False
                previous_word_idx = idx
            # Apply penalty based on edit count
            if pred_probs is None and edit_count > 0:
                penalty = 1.0 - 0.1 * min(edit_count, 4)  # Stronger penalty for more edits
                conf_scores = [score * penalty for score in conf_scores]
            pred_labels.append(labels)
            no_corrections.append(no_correct)
            confidences.append(conf_scores)
    
    return pred_labels, no_corrections, confidences

def predict(
    model: GECToR,
    tokenizer: PreTrainedTokenizer,
    srcs: List[str],
    encode: dict,
    decode: dict,
    keep_confidence: float = 0,
    min_error_prob: float = 0,
    batch_size: int = 128,
    n_iteration: int = 5
) -> Tuple[List[str], List[float]]:
    """
    Predict corrected sentences and their confidence scores.
    
    Args:
        model: GECToR model
        tokenizer: PreTrainedTokenizer
        srcs: List of input sentences
        encode: Dictionary for verb transformations
        decode: Dictionary for verb transformations
        keep_confidence: Confidence threshold for keeping predictions
        min_error_prob: Minimum error probability threshold
        batch_size: Batch size for processing
        n_iteration: Maximum number of correction iterations
    
    Returns:
        Tuple of (corrected sentences, confidence scores as mean of per-token scores)
    """
    if not srcs:
        logger.warning("Empty input list provided to predict")
        return [], []
    
    logger.info(f"Processing {len(srcs)} sentences with up to {n_iteration} iterations")
    
    # Initialize input with $START token
    srcs = [['$START'] + src.split(' ') for src in srcs]
    final_edited_sents = ['-1'] * len(srcs)
    final_conf_scores = [0.0] * len(srcs)
    to_be_processed = srcs
    original_sent_idx = list(range(len(srcs)))
    
    for itr in tqdm(range(n_iteration), desc="Correction iterations"):
        logger.info(f"Iteration {itr}. Sentences to process: {len(to_be_processed)}")
        pred_labels, no_corrections, confidences = _predict(
            model,
            tokenizer,
            to_be_processed,
            keep_confidence,
            min_error_prob,
            batch_size
        )
        current_srcs = []
        current_pred_labels = []
        current_confidences = []
        current_orig_idx = []
        
        for i, (yes, conf) in enumerate(zip(no_corrections, confidences)):
            if yes:
                final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
                final_conf_scores[original_sent_idx[i]] = sum(conf) / len(conf) if conf else 0.95
            else:
                current_srcs.append(to_be_processed[i])
                current_pred_labels.append(pred_labels[i])
                current_confidences.append(conf)
                current_orig_idx.append(original_sent_idx[i])
        
        if not current_srcs:
            logger.info("All sentences corrected. Exiting early.")
            break
        
        edited_srcs = edit_src_by_tags(
            current_srcs,
            current_pred_labels,
            encode,
            decode
        )
        to_be_processed = edited_srcs
        original_sent_idx = current_orig_idx
        confidences = current_confidences
    
    # Finalize remaining sentences
    for i in range(len(to_be_processed)):
        final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
        final_conf_scores[original_sent_idx[i]] = sum(confidences[i]) / len(confidences[i]) if confidences[i] else 0.95
    
    assert '-1' not in final_edited_sents, "Some sentences were not processed"
    
    logger.info("Prediction completed")
    return final_edited_sents, final_conf_scores
