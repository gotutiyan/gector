import argparse
from gector import (
    GECToR,
    predict,
    load_verb_dict
)
from transformers import AutoTokenizer
import torch
import os
import numpy as np

def main(args):
    output_path = os.path.join(args.restore_dir, 'outputs', 'tweak_outputs')
    os.makedirs(output_path, exist_ok=True)
    model = GECToR.from_pretrained(args.restore_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    srcs = open(args.input).read().rstrip().split('\n')
    encode, decode = load_verb_dict(args.verb_file)
    if torch.cuda.is_available():
        model.cuda()
    for kc in np.arange(args.kc_min, args.kc_max, args.step):
        for mep in np.arange(args.mep_min, args.mep_max, args.step):
            final_corrected_sents = predict(
                model,
                tokenizer,
                srcs,
                encode,
                decode,
                keep_confidence=kc,
                min_error_prob=mep,
                batch_size=args.batch_size,
                n_iteration=args.n_iteration
            )
            with open(os.path.join(output_path, f'kc{str(round(kc, 1))}_mep{str(round(mep, 1))}.txt'), 'w') as f:
                f.write('\n'.join(final_corrected_sents))

def cli_main():
    args = get_parser()
    main(args)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--verb_file', default='data/verb-form-vocab.txt')
    parser.add_argument('--n_iteration', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--kc_min', type=float, default=0)
    parser.add_argument('--kc_max', type=float, default=1)
    parser.add_argument('--mep_min', type=float, default=0)
    parser.add_argument('--mep_max', type=float, default=1)
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--out', default='out.txt')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)