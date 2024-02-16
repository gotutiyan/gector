# GECToR

This is one of the implementation of the following [paper](https://aclanthology.org/2020.bea-1.16.pdf):
```
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn  and
      Atrasevych, Vitaliy  and
      Chernodub, Artem  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA → Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.bea-1.16",
    doi = "10.18653/v1/2020.bea-1.16",
    pages = "163--170"
}
```

# Differences from other implementations
- Official: [grammarly/gector](https://github.com/grammarly/gector)
    - Without AllenNLP
    - Trained checkpoints can be downloaded from Hub
    - Distributed training
    - 😔 Does not support probabilistic ensemble
- [cofe-ai/fast-gector](https://github.com/cofe-ai/fast-gector)
    - Use Accelerate for distributed training

# Installing
Confirmed that it works on python3.11.0.
```sh
pip install -r requirements.txt
# Donwload the verb dictionary in advance
mkdir data
cd data
wget https://github.com/grammarly/gector/raw/master/data/verb-form-vocab.txt
```

# Usage
- I will published pre-trained weights on Hugging Face Hub. Please refer to [Performances obtained](https://github.com/gotutiyan/gector#performances_obtained).

- Note that this implementation does not support probabilistic ensembling. See [Ensemble](https://github.com/gotutiyan/gector#ensemble).
 
CLI
```sh
python predict.py \
    --input <raw text file> \
    --restore_dir gotutiyan/gector-roberta-base-5k \
    --out <path to output file>
```

API
```py
from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict

model_id = 'gotutiyan/gector-roberta-base-5k'
model = GECToR.from_pretrained(model_id)
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

# Performances obtained

I performed experiments using this implementation. Trained models are also obtained from Hugging Face Hub.

<details>
<summary>The details of experimental settings: </summary>

- All models below are trained on all of stages 1, 2, and 3.

### Configurations
- The common training config is the following:
```json
{
    "restore_vocab_official": "data/output_vocabulary/",
    "max_len": 80,
    "n_epochs": 10,
    "p_dropout": 0.0,
    "lr": 1e-05,
    "cold_lr": 0.001,
    "accumulation": 1,
    "label_smoothing": 0.0,
    "num_warmup_steps": 500,
    "lr_scheduler_type": "constant"
}
```

For stage1, 
```json
{
    "batch_size": 256,
    "n_cold_epochs": 2
}
```
For stage2, 
```json
{
    "batch_size": 128,
    "n_cold_epochs": 2
}
```
For stage3,
```json
{
    "batch_size": 128,
    "n_cold_epochs": 0
}
```

### Datasets

|Stage|Train Datasets (# sents.)|Validation Dataset (# sents.)|
|:-:|:--|:--|
|1|PIE-synthetic (8,865,347, a1 split of [this](https://drive.google.com/file/d/1bl5reJ-XhPEfEaPjvO45M7w0yN-0XGOA/view))|BEA19-dev (i.e. W&I+LOCNESS-dev, 4,382)|
|2|BEA19-train: FCE-train + W&I+LOCNESS-train + Lang-8 + NUCLE, without src=trg pairs (561,290)|BEA19-dev|
|3|W&I+LOCNESS-train (34,304)|BEA19-dev|

- Note that the number of epochs for stage1 is smaller than official setting (= 20 epochs). The reasons for this are (1) the results were competitive the results in the paper even at 10 epochs, and (2) I did not want to occupy as much computational resources in my laboratory as possible.
- The tag vocabulary is the same as [official one](https://github.com/grammarly/gector/blob/master/data/output_vocabulary/labels.txt).
- I trained on three different seeds (10,11,12) for each model, and use the one with the best performance.
    - Futhermore, I tweaked a keep confidence and a sentence-level minimum error probability threshold (from 0 to 0.9, 0.1 steps each) for each best model. 
    - Finally, the checkpoint with the highest F0.5 on BEA19-dev is used. 
    - The number of iterations is 5.

### Evaluation

- Used ERRANT for the BEA19-dev evaluation.
    - I merely used official reference M2 file for the evaluation. Basically, the edit spans of reference M2 should be obtained again with ERRANT (`errant_m2 -auto`). However, I do not know if many research do that, and it seems that they do not. Thus I also do not that.
- Used [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/4057) for the BEA19-test evaluation.
- Used M2 Scorer for the CoNLL14 evaluation.

</details>




### Single setting

#### Base-5k
|Model|Confidence|Threshold|BEA19-dev (P/R/F0.5)|CoNLL14 (P/R/F0.5)|BEA19-test (P/R/F0.5)|
|:--|:-:|:-:|:-:|:-:|:-:|
|BERT [[Omelianchuk+ 2020]](https://aclanthology.org/2020.bea-1.16/)||||72.1/42.0/63.0|71.5/55.7/67.6|
|RoBERTa [[Omelianchuk+ 2020]](https://aclanthology.org/2020.bea-1.16/)||||73.9/41.5/64.0|77.2/55.1/71.5|
|XLNet [[Omelianchuk+ 2020]](https://aclanthology.org/2020.bea-1.16/)|||66.0/33.8/55.5|77.5/40.1/65.3|79.2/53.9/72.4|
|DeBERTa [[Tarnavskyi+ 2022]](https://aclanthology.org/2022.acl-long.266/)(Table 3)|||64.2/31.8/53.8|||
|[gotutiyan/gector-bert-base-cased-5k](https://huggingface.co/gotutiyan/gector-bert-base-cased-5k)|0.4|0.6|64.5/30.0/52.4|73.0/33.6/59.1|76.8/48.7/68.9|
|[gotutiyan/gector-roberta-base-5k](https://huggingface.co/gotutiyan/gector-roberta-base-5k)|0.5|0.0|65.8/31.8/54.2|74.6/35.7/61.3|78.5/51.0/70.8|
|[gotutiyan/gector-xlnet-base-cased-5k](https://huggingface.co/gotutiyan/gector-xlnet-base-cased-5k)|0.5|0.0|67.2/30.7/54.3|77.2/34.4/61.8|78.8/49.9/70.7|
|[gotutiyan/gector-deberta-base-5k](https://huggingface.co/gotutiyan/gector-deberta-base-5k)|0.4|0.3|64.1/34.5/54.7|73.7/38.8/62.5|76.0/54.2/70.4|

#### Large-5k
|Model|Confidence|Threshold|BEA19-dev (P/R/F0.5)|CoNLL14 (P/R/F0.5)|BEA19-test (P/R/F0.5)|
|:--|:-:|:-:|:-:|:-:|:-:|
|RoBERTa [[Tarnavskyi+ 2022]](https://aclanthology.org/2022.acl-long.266/)|||65.7/33.8/55.3||80.7/53.3/73.2|
|XLNet [[Tarnavskyi+ 2022]](https://aclanthology.org/2022.acl-long.266/)|||64.2/35.1/55.1|||
|DeBERTa [[Tarnavskyi+ 2022]](https://aclanthology.org/2022.acl-long.266/)|||66.3/32.7/55.0|||
|DeBERTa (basetag) [[Mesham+ 2023]](https://aclanthology.org/2023.findings-eacl.119)|||68.1/38.1/58.8||77.8/56.7/72.4|
|[gotutiyan/gector-bert-large-cased-5k](https://huggingface.co/gotutiyan/gector-bert-large-cased-5k)|0.5|0.0|64.7/32.0/53.7|75.9/36.8/62.6|77.2/50.4/69.8|
|[gotutiyan/gector-roberta-large-5k](https://huggingface.co/gotutiyan/gector-roberta-large-5k)|0.4|0.6|65.7/34.3/55.5|75.4/37.1/62.5|78.5/53.7/71.9|
|[gotutiyan/gector-xlnet-large-cased-5k](https://huggingface.co/gotutiyan/gector-xlnet-large-cased-5k)|0.3|0.4|63.8/36.5/55.5|74.6/41.6/64.4|75.9/56.7/71.1|
|[gotutiyan/gector-deberta-large-5k](https://huggingface.co/gotutiyan/gector-deberta-large-5k)|0.5|0.4|68.7/33.1/56.6|80.0/36.9/64.8|81.1/52.8/73.2|

### Ensemble setting
|Model|CoNLL14 (P/R/F0.5)|BEA19-test (P/R/F0.5)|Note|
|:--|:-:|:-:|:--|
|BERT(base) + RoBERTa(base) + XLNet(base) [[Omelianchuk+ 2020]](https://aclanthology.org/2020.bea-1.16/)|78.2/41.5/66.5|78.9/58.2/73.6||
|gotutiyan/gector-bert-base-cased-5k + gotutiyan/gector-roberta-base-5k + gotutiyan/gector-xlnet-base-cased-5k|80.9/33.3/63.0|83.5/48.7/73.1|The ensemble method is different from [Omelianchuk+ 2020](https://aclanthology.org/2020.bea-1.16/).|
|RoBERTa(large, 10k) + XLNet(large, 5k) + DeBERTa(large, 10k) [[Tarnavskyi+ 2022]](https://aclanthology.org/2022.acl-long.266/)||84.4/54.4/76.0||
|gotutiyan/gector-roberta-large-5k + gotutiyan/gector-xlnet-large-cased-5k + gotutiyan/gector-deberta-large-5k|81.7/37.0/65.8|84.0/53.4/75.4|


# How to train
### Preprocess
Use official preprocessing code.
E.g.
```sh
mkdir utils
cd utils
wget https://github.com/grammarly/gector/raw/master/utils/preprocess_data.py
wget https://raw.githubusercontent.com/grammarly/gector/master/utils/helpers.py
cd ..
python utils/preprocess_data.py \
    -s <raw source file path> \
    -t <raw target file path> \
    -o <output path>
```

### Train
`train.py` uses Accelerate. Please input your environment with `accelerate config` in advance.
```sh
accelerate launch train.py \
    --train_file <preprocess output of train> \
    --valid_file <preprocess output of validation> \
    --save_dir outputs/sample
```

<details>
<summary>Other options of train.py :</summary>

|Option|Default|Note|
|:--|:--|:--|
|--model_id|bert-base-cased|Specify BERT-like model. I confirmed that `bert-**`, `roberta-**`, `microsoft/deberta-`, `xlnet-**` are worked.|
|--batch_size|16||
|--delimeter|`SEPL\|\|\|SEPR`|The delimeter of preprocessed file.|
|--additional_delimeter|`SEPL__SEPR`|Another delimeter to split multiple tags for a word.|
|--restore_dir|None|For training from specified checkpoint. Both weights and tag vocab will be loaded.|
|--restore_vocab|None|To train with existing tag vocabulary. Please specify `config.json` to this. Note that weights are not loaded.|
|--restore_vocab_official|None|Use existing tag vocabulary in the official format. Please specify like `path/to/data/output_vocabulary/`|
|--max_len|80|Maximum length of input (subword-level length)|
|--n_max_labels|5000|The number of tag types.|
|--n_epochs|10|The number of epochs.|
|--n_cold_epochs|2|The number of epochs to train only classifier layer.|
|--lr|1e-5|The learning rate after cold steps.|
|--cold_lr|1e-3|The learning rate during cold steps.|
|--p_dropout|0.0|The dropout rate of label projection layers.|
|--accumulation|1|The number of accumulation.|
|--seed|10|seed|
|--label_smoothing|0.0|The label smoothing of the CrossEntropyLoss.|
|--num_warmup_steps|500|The number of warmup for learning rate scheduler.|
|--lr_scheduler_type|constant|Specify leaning rate scheduler type.|

NOTE: For those who are familiar with the [official implementation](https://github.com/grammarly/gector/tree/master),
- `--tag_strategy` is not available but it is forced to keep_one.
- `--skip_correct` is not available. Please remove identical pairs from your training data in advance.
- `--patience` is not available since this implementation does not employ early stopping.
- `--special_token_fix` is not available since this code automatically judge this one from `--model_id`.

</details>

The best and last checkpoints are saved. The format is:
```
outputs/sample
├── best
│   ├── added_tokens.json
│   ├── config.json
│   ├── merges.txt
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── last
│   ├── ... (The same as best/)
└── log.json
```

# Inference

The same usage of the [Usage](https://github.com/gotutiyan/gector#usage). You can specify `best/` or `last/` directory to `--restore_dir`.

CLI
```sh
python predict.py \
    --input <raw text file> \
    --restore_dir outputs/sample/best \
    --out <path to output file>
```
<details>
<summary>Other options of predict.py: </summary>

|Option|Default|Note|
|:--|:--|:--|
|--n_iteration|5|The number of iterations.|
|--batch_size|128|A Batch size.|
|--keep_confidence|0.0|A bias for the $KEEP label.|
|--min_error_prob|0.0|A sentence-level minimum error
probability threshold|
|--verb_file|`data/verb-form-vocab.txt`|Assume that you already have this file by [Installing]((https://github.com/gotutiyan/gector#installing)).|
|--visualize|None|Output visualization results to a specified file.|

</details>

Or, to use as API,
```py
from transformers import AutoTokenizer
from gector import GECToR

path = 'outputs/sample/best'
model = GECToR.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
```

### Visualize the predictions

You can use `--visualize` option to output a visualization of the predictions. It will be helpful for qualitative analyses.

For example,
```sh
echo 'A ten years old boy go school' > demo.txt
python predict.py \
--restore_dir gotutiyan/gector-roberta-base-5k \
--input demo.txt \
--visualize visualize.txt
```

`visualize.txt` will show:
```
=== Line 0 ===
== Iteration 0 ==
|$START |A     |ten       |years                         |old   |boy   |go                     |school |
|$KEEP  |$KEEP |$APPEND_- |$TRANSFORM_AGREEMENT_SINGULAR |$KEEP |$KEEP |$TRANSFORM_VERB_VB_VBZ |$KEEP  |
== Iteration 1 ==
|$START |A     |ten   |-     |year  |old   |boy   |goes       |school |
|$KEEP  |$KEEP |$KEEP |$KEEP |$KEEP |$KEEP |$KEEP |$APPEND_to |$KEEP  |
== Iteration 2 ==
|$START |A     |ten   |-     |year      |old   |boy   |goes  |to    |school |
|$KEEP  |$KEEP |$KEEP |$KEEP |$APPEND_- |$KEEP |$KEEP |$KEEP |$KEEP |$KEEP  |
A ten - year - old boy goes to school
```


### Tweak parameters

To tweak two parameters in the inference, please use `predict_tweak.py`.  
The following example tweaks both of parameters in `{0, 0.1, 0.2 ... 0.9}`. `kc` is a keep confidence and `mep` is a minimum error probability threshold.
```sh
python predict_tweak.py \
    --input <raw text file> \
    --restore_dir outputs/sample/best \
    --kc_min 0 \
    --kc_max 1 \
    --mep_min 0 \
    --mep_max 1 \
    --step 0.1
```

This script creates `<--restore_dir>/outputs/tweak_outputs/` and saves each output in it.
```
models/sample/best/outputs/tweak_outputs/
├── kc0.0_mep0.0.txt
├── kc0.0_mep0.1.txt
├── kc0.0_mep0.2.txt
...
```

After that, you can determine the best parameters by doing the following:
```sh
RESTORE_DIR=${1}
for kc in `seq 0 0.1 0.9` ; do
for mep in `seq 0 0.1 0.9` ; do
# Refer to $RESTORE_DIR/outputs/tweak_output/kc${kc}_mep${mep}.txt in the evaluation scripts
done
done
```

### Ensemble

- This implementation does not support probabilistic ensemble inference. Please use majority voting ensemble [[Tarnavskyi+ 2022]](https://aclanthology.org/2022.acl-long.266/) instead.
```sh
wget https://github.com/MaksTarnavskyi/gector-large/raw/master/ensemble.py
python ensemble.py \
    --source_file <source> \
    --target_files <hyp1> <hyp2> ... \
    --output_file <out>
```