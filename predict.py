import argparse
from gector.modeling import GECToR
from transformers import AutoTokenizer
from gector.predict import predict, load_verb_dict
import torch

def main(args):
    if args.test:
        test()
        return
    model = GECToR.from_pretrained(args.restore_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    srcs = open(args.input).read().rstrip().split('\n')
    encode, decode = load_verb_dict(args.verb_file)
    if torch.cuda.is_available():
        model.cuda()
    final_corrected_sents = predict(
        model,
        tokenizer,
        srcs,
        encode,
        decode,
        keep_confidence=args.keep_confidence,
        min_error_prob=args.min_error_prob,
        batch_size=args.batch_size,
        n_iteration=args.n_iteration
    )
    with open(args.out, 'w') as f:
        f.write('\n'.join(final_corrected_sents))
    print(f'=== Finished ===')

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
    parser.add_argument('--input')
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--verb_file', default='data/verb-form-vocab.txt')
    parser.add_argument('--n_iteration', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--keep_confidence', type=float, default=0)
    parser.add_argument('--min_error_prob', type=float, default=0)
    parser.add_argument('--out', default='out.txt')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)