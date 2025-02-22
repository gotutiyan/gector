import argparse
from gector import (
    GECToR,
    predict,
    load_verb_dict,
    predict_verbose
)
from transformers import AutoTokenizer
import torch
from typing import List, Dict

def need_add_prefix_space(model_id):
    for m in ['roberta', 'deberta']:
        if m in model_id:
            return True
    return False

def visualizer(iteration_log: List[List[Dict]]):
    # Generate a string to visualize the predictions.
    strs = ''
    for sent_id, sent in enumerate(iteration_log):
        strs += f'=== Line {sent_id} ===\n'
        for itr_id, itr in enumerate(sent):
            if itr['tag'] is None:
                strs += ' '.join(itr['src']).replace('$START', '').strip() + '\n'
                break
            strs += f'== Iteration {itr_id} ==\n'
            src_str = '|'
            tag_str = '|'
            for tok, tag in zip(itr['src'], itr['tag']):
                max_len = max(len(tok), len(tag)) + 1
                src_str += tok + ' '*(max_len - len(tok)) + '|'
                tag_str += tag + ' '*(max_len - len(tag)) + '|'
            strs += src_str + '\n'
            strs += tag_str + '\n'
        strs += '\n'
    return strs
                
def main(args):
    if args.test:
        test()
        return
    if args.from_official:
        # Use official weights.
        model = GECToR.from_official_pretrained(
            args.restore_dir,
            special_tokens_fix=getattr(args, 'official.special_tokens_fix'),
            transformer_model=getattr(args, 'official.transformer_model'),
            vocab_path=getattr(args, 'official.vocab_path'),
            max_length=getattr(args, 'official.max_length')
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            getattr(args, 'official.transformer_model'),
            add_prefix_space=need_add_prefix_space(getattr(args, 'official.transformer_model'))
        )
        if getattr(args, 'official.special_tokens_fix'):
            # if special_tokens_fix is 1, the official model was trained 
            #   by adding $START token. So we add $START to the tokenizer.
            tokenizer.add_special_tokens(
                {'additional_special_tokens': ['$START']}
            )
    else:
        model = GECToR.from_pretrained(args.restore_dir).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    srcs = open(args.input).read().rstrip().split('\n')
    encode, decode = load_verb_dict(args.verb_file)
    if torch.cuda.is_available():
        model.cuda()
    predict_args = {
        'model': model,
        'tokenizer': tokenizer,
        'srcs': srcs,
        'encode': encode,
        'decode': decode,
        'keep_confidence': args.keep_confidence,
        'min_error_prob': args.min_error_prob,
        'batch_size': args.batch_size,
        'n_iteration': args.n_iteration
    }
    if args.visualize is not None:
        final_corrected_sents, iteration_log = predict_verbose(
            **predict_args
        )
        strs = visualizer(iteration_log)
        with open(args.visualize, 'w') as fp:
            fp.write(strs)
    else:
        final_corrected_sents = predict(
            **predict_args
        )
    with open(args.out, 'w') as f:
        f.write('\n'.join(final_corrected_sents))

def cli_main():
    args = get_parser()
    main(args)

def test():
    data = [
        ['There flowers .', # source
        '$APPEND_are $KEEP $KEEP', # tags
        'There are flowers .' # reference
        ],
        ['There flowers',
        '$APPEND_are $APPEND_.',
        'There are flowers .'
        ],
        ['dummy There are a flowers . dummy',
        '$DELETE $KEEP $KEEP $DELETE $KEEP $KEEP $DELETE',
        'There are flowers .'
        ],
        ['There is flowers .',
        '$KEEP $REPLACE_are $KEEP $KEEP',
        'There are flowers .'
        ],
        ['There are flo wers .',
        '$KEEP $KEEP $MERGE_SPACE $KEEP, $KEEP',
        'There are flowers .'
        ],
        ['Position wise network',
        '$MERGE_HYPHEN $KEEP $KEEP',
        'Position-wise network'
        ],
        ['I like iphone .',
        '$KEEP $KEEP $TRANSFORM_CASE_CAPITAL_1 $KEEP',
        'I like iPhone .'
        ],
        ['there ARE many iphone .',
        '$TRANSFORM_CASE_CAPITAL $TRANSFORM_CASE_LOWER $TRANSFORM_CASE_UPPER $TRANSFORM_CASE_CAPITAL_1 $KEEP',
        'There are MANY iPhone .'
        ],
        ['There are flower .',
        '$KEEP $KEEP $TRANSFORM_AGREEMENT_PLURAL $KEEP',
        'There are flowers .'
        ],
        ['There is a flowers .',
        '$KEEP $KEEP $KEEP $TRANSFORM_AGREEMENT_SINGULAR $KEEP',
        'There is a flower .'
        ],
        ['There are many-flowers .',
        '$KEEP $KEEP $TRANSFORM_SPLIT_HYPHEN $KEEP',
        'There are many flowers .'
        ],
        ['He go to school .',
        '$KEEP $TRANSFORM_VERB_VB_VBZ $KEEP $KEEP $KEEP',
        'He goes to school .'
        ],
        ['He go to school yesterday .',
        '$KEEP $TRANSFORM_VERB_VB_VBD $KEEP $KEEP $KEEP',
        'He went to school yesterday .'
        ],
        ['The letter is write in Japanese .',
        '$KEEP $KEEP $KEEP $TRANSFORM_VERB_VB_VBN $KEEP $KEEP $KEEP',
        'The letter is written in Japanese .'
        ],
        ['I goes to school .',
        '$KEEP $TRANSFORM_VERB_VBZ_VB $KEEP $KEEP $KEEP',
        'I go to school .'
        ],
    ]
    from gector.predict import edit_src_by_tags
    stoken = '$START'
    srcs = [[stoken] + item[0].split() for item in data]
    labels = [['$KEEP'] + item[1].split() for item in data]
    refs = [stoken + ' ' + item[2] for item in data]
    encode, decode = load_verb_dict('data/verb-form-vocab.txt')
    edited_srcs = edit_src_by_tags(
        srcs,
        labels,
        encode, decode
    )
    for hyp, ref in zip(edited_srcs, refs):
        # print(' '.join(hyp), ref)
        assert(' '.join(hyp) == ref)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--verb_file', default='data/verb-form-vocab.txt')
    parser.add_argument('--n_iteration', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--keep_confidence', type=float, default=0)
    parser.add_argument('--min_error_prob', type=float, default=0)
    parser.add_argument('--out', default='out.txt')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--visualize')
    parser.add_argument(
        '--from_official', action='store_true',
        help='Specify this if you load official weights.'
    )
    parser.add_argument(
        '--official.vocab_path', default='data/output_vocabulary',
        help='The vocabulary directory when using official model.'
    )
    parser.add_argument(
        '--official.transformer_model', default='bert-base-cased',
        help='The model id of HF trasnformers when using official model.'
    )
    parser.add_argument(
        '--official.special_tokens_fix', type=int, default=0,
        help='0 or 1 according to the training setting of the official model.'
    )
    parser.add_argument(
        '--official.max_length', type=int, default=80,
        help='If the number of subwords is longer than this, it will be truncated.'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)