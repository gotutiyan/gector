import argparse
from huggingface_hub import create_repo, upload_folder, ModelCard
def main(args):
    create_repo(repo_id=args.repo_id, exist_ok=True, repo_type='model')
    upload_folder(
        folder_path=args.dir,
        repo_id=args.repo_id,
        repo_type="model",
    )
    content = f"""
---
language: en
license: mit
tags:
- GECToR_gotutiyan
---

# gector sample
This is an unofficial pretrained model of GECToR ([Omelianchuk+ 2020](https://aclanthology.org/2020.bea-1.16/)).

### How to use
The code is avaliable from https://github.com/gotutiyan/gector.

CLI
```sh
python predict.py --input <raw text file> \
    --restore_dir {args.repo_id} \
    --out <path to output file>
```

API
```py
from transformers import AutoTokenizer
from gector.modeling import GECToR
from gector.predict import predict, load_verb_dict
import torch

model_id = '{args.repo_id}'
model = GECToR.from_pretrained(model_id)
if torch.cuda.is_available():
    model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
encode, decode = load_verb_dict('data/verb-form-vocab.txt')
srcs = [
    'This is a correct sentence.',
    'This are a wrong sentences'
]
corrected = predict(
    model, tokenizer, srcs,
    encode, decode,
    keep_confidence=0.0,
    min_error_prob=0.0,
    n_iteration=5,
    batch_size=2,
)
print(corrected)
```
"""
    card = ModelCard(content)
    # print(card.data.to_dict())
    card.push_to_hub(args.repo_id)

    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', required=True)
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)