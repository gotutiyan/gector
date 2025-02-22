import argparse
from transformers import AutoTokenizer, get_scheduler
from gector import (
    GECToR,
    GECToRConfig,
    load_dataset,
    build_vocab,
    load_vocab_from_config,
    load_vocab_from_official
)
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import os
from tqdm import tqdm
import json
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import numpy as np
import random
from collections import OrderedDict
from gector.utils import has_args_add_pooling

def solve_model_id(model_id):
    if model_id == 'deberta-base':
        return 'microsoft/deberta-base'
    elif model_id == 'deberta-large':
        return 'microsoft/deberta-large'
    else:
        return model_id

def train(
    model,
    loader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
    step_scheduler
):
    log = {
        'loss': 0,
        'accuracy': 0,
        'accuracy_d': 0
    }
    model.train()
    pbar = tqdm(loader, total=len(loader), disable=not accelerator.is_main_process)
    for batch in pbar:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if step_scheduler:
                lr_scheduler.step()
            log['loss'] += loss.item()
            log['accuracy'] += outputs.accuracy.item()
            log['accuracy_d'] += outputs.accuracy_d.item()
            if accelerator.is_main_process:
                pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
                pbar.set_postfix(OrderedDict(
                    loss=loss.item(),
                    accuracy=outputs.accuracy.item(),
                    accuracy_d=outputs.accuracy_d.item(),
                    lr=optimizer.param_groups[0]['lr']
                ))
    return {k:v/len(loader) for k,v in log.items()}

@torch.no_grad()
def valid(
    model,
    loader,
    accelerator,
    epoch
):
    log = {
        'loss': 0,
        'accuracy': 0,
        'accuracy_d': 0
    }
    model.eval()
    pbar = tqdm(loader, total=len(loader), disable=not accelerator.is_main_process)
    for batch in pbar:
        outputs = model(**batch)
        log['loss'] += outputs.loss.item()
        log['accuracy'] += outputs.accuracy.item()
        log['accuracy_d'] += outputs.accuracy_d.item()
        if accelerator.is_main_process:
            pbar.set_description(f'[Epoch {epoch}] [VALID]')
            pbar.set_postfix(OrderedDict(
                loss=outputs.loss.item(),
                accuracy=outputs.accuracy.item(),
                accuracy_d=outputs.accuracy_d.item(),
            ))
    return {k:v/len(loader) for k,v in log.items()}

def main(args):
    # To easily specify the model_id 
    args.model_id = solve_model_id(args.model_id)
    print('Start ...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    if args.restore_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            add_prefix_space=True
        )
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['$START']}
    )

    print('Loading datasets...')
    dataset_args = {
        'input_file': args.train_file,
        'tokenizer': tokenizer,
        'delimeter': args.delimeter,
        'additional_delimeter': args.additional_delimeter,
        'max_length': args.max_len
    }
    train_dataset = load_dataset(**dataset_args)
    dataset_args['input_file'] = args.valid_file
    valid_dataset = load_dataset(**dataset_args)
    if args.restore_dir is not None:
        # If you specify path or id to --restore_dir, the model loads weights and vocab.
        model = GECToR.from_pretrained(args.restore_dir)
    else:
        # Otherwise, the model will be trained from scratch.
        if args.restore_vocab is not None:
            # But you can use existing vocab.
            label2id, d_label2id = load_vocab_from_config(args.restore_vocab)
        elif args.restore_vocab_official is not None:
            label2id, d_label2id = load_vocab_from_official(args.restore_vocab_official)
        else:
            print('Builing vocab...')
            label2id, d_label2id = build_vocab(
                train_dataset,
                n_max_labels=args.n_max_labels,
                n_max_d_labels=2
            )
        gector_config = GECToRConfig(
            model_id=args.model_id,
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
            d_label2id=d_label2id,
            p_dropout=args.p_dropout,
            max_length=args.max_len,
            label_smoothing=args.label_smoothing,
            has_add_pooling_layer=has_args_add_pooling(args.model_id)
        )
        model = GECToR(config=gector_config)
    train_dataset.append_vocab(
        model.config.label2id,
        model.config.d_label2id
    )
    valid_dataset.append_vocab(
        model.config.label2id,
        model.config.d_label2id
    )
    print('# instances of train:', len(train_dataset))
    print('# instances of valid:', len(valid_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * (args.n_epochs - args.n_cold_epochs) // args.accumulation,
    )
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )
    path_to_best = os.path.join(args.save_dir, 'best')
    path_to_last = os.path.join(args.save_dir, 'last')
    os.makedirs(path_to_best, exist_ok=True)
    os.makedirs(path_to_last, exist_ok=True)
    tokenizer.save_pretrained(path_to_best)
    tokenizer.save_pretrained(path_to_last)
    max_acc = -1
    print('Start training...')
    def set_lr(optimizer, lr):
        for param in optimizer.param_groups:
            param['lr'] = lr
    logs = {'argparse': args.__dict__}
    for e in range(args.n_epochs):
        accelerator.wait_for_everyone()
        if isinstance(model, DistributedDataParallel):
            module = model.module
        else:
            module = model
        step_scheduler = True
        if e < args.n_cold_epochs:
            module.tune_bert(False)
            set_lr(optimizer, args.cold_lr)
            step_scheduler = False
        elif e == args.n_cold_epochs:
            module.tune_bert(True)
            set_lr(optimizer, args.lr)
        else:
            pass
        print(f'=== Epoch {e} ===')
        train_log = train(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            accelerator,
            e,
            step_scheduler
        )
        valid_log = valid(
            model,
            valid_loader,
            accelerator,
            e
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if valid_log['accuracy'] > max_acc:
                accelerator.unwrap_model(model).save_pretrained(path_to_best)
                max_acc = valid_log['accuracy']
                valid_log['message'] = 'The best checkpoint has been updated.'
            accelerator.unwrap_model(model).save_pretrained(path_to_last)
            logs[f'Epoch {e}'] = {
                'train_log': train_log,
                'valid_log': valid_log
            }
            with open(os.path.join(args.save_dir, 'log.json'), 'w') as f:
                json.dump(logs, f, indent=2)
    print('finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--valid_file', required=True)
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--delimeter', default='SEPL|||SEPR')
    parser.add_argument('--additional_delimeter', default='SEPL__SEPR')
    parser.add_argument('--restore_dir')
    parser.add_argument('--restore_vocab')
    parser.add_argument('--restore_vocab_official')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--n_max_labels', type=int, default=5000)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--p_dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cold_lr', type=float, default=1e-3)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--n_cold_epochs', type=int, default=2)
    parser.add_argument('--num_warmup_steps', type=int, default=500)
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)

# bert-base-cased roberta-base deberta-base xlnet-base-cased
# bert-large-cased roberta-large deberta-large xlnet-large-cased