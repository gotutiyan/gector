from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from .configuration import GECToRConfig
from typing import List
import os
import json
from huggingface_hub import snapshot_download, ModelCard

@dataclass
class GECToROutput:
    loss: torch.Tensor = None
    loss_d: torch.Tensor = None
    loss_labels: torch.Tensor = None
    logits_d: torch.Tensor = None
    logits_labels: torch.Tensor = None
    accuracy: torch.Tensor = None
    accuracy_d: torch.Tensor = None

@dataclass
class GECToRPredictionOutput:
    probability_labels: torch.Tensor = None
    probability_d: torch.Tensor = None
    pred_labels: List[List[str]] = None
    pred_label_ids: torch.Tensor = None
    max_error_probability: torch.Tensor = None

class GECToR(nn.Module):
    def __init__(
        self,
        config: GECToRConfig=None
    ):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if self.has_add_pooling_layer(self.config.model_id):
            self.bert = AutoModel.from_pretrained(
                self.config.model_id,
                add_pooling_layer=False
            )
        else:
            self.bert = AutoModel.from_pretrained(self.config.model_id)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 1) # +1 is for $START token
        self.label_proj_layer = nn.Linear(
            self.bert.config.hidden_size,
            config.n_labels-1
        ) # -1 is for <PAD>
        self.d_proj_layer = nn.Linear(
            self.bert.config.hidden_size,
            config.n_d_labels-1
        )
        self.dropout = nn.Dropout(self.config.p_dropout)
        self.loss_fn = CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        self._init_weights(self.label_proj_layer)
        self._init_weights(self.d_proj_layer)
        self.tune_bert(False)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        return

    def tune_bert(self, tune=True):
        # If tune=False, only classifier layers will be tuned.
        for param in self.bert.parameters():
            param.requires_grad = tune
        return

    @staticmethod
    def has_add_pooling_layer(model_id):
        for id in ['xlnet', 'deberta']:
            if id in model_id:
                return False
        return True
    
    def forward(
        self,
        input_ids,
        attention_mask,
        d_labels=None,
        labels=None,
        word_masks=None,
        **kwards
    ):
        bert_logits = self.bert(
            input_ids,
            attention_mask
        ).last_hidden_state
        logits_d = self.d_proj_layer(bert_logits)
        logits_labels = self.label_proj_layer(self.dropout(bert_logits))
        loss_d, loss_labels, loss = None, None, None
        accuracy, accuracy_d = None, None
        if d_labels is not None and labels is not None:
            pad_id = self.config.label2id[self.config.label_pad_token]
            labels[labels == pad_id] = -100 # -100 is the default ignore_idx of CrossEntropyLoss
            d_labels[labels == -100] = -100
            loss_d = self.loss_fn(
                logits_d.view(-1, self.config.n_d_labels-1), # -1 for <PAD>
                d_labels.view(-1)
            )
            loss_labels = self.loss_fn(
                logits_labels.view(-1, self.config.n_labels-1),
                labels.view(-1)
            )
            loss = loss_d + loss_labels

            pred_labels = torch.argmax(logits_labels, dim=-1)
            accuracy = torch.sum((labels == pred_labels) * word_masks) / torch.sum(word_masks)
            pred_d = torch.argmax(logits_d, dim=-1)
            accuracy_d = torch.sum((d_labels == pred_d) * word_masks) / torch.sum(word_masks)

        return GECToROutput(
            loss=loss,
            loss_d=loss_d,
            loss_labels=loss_labels,
            logits_d=logits_d,
            logits_labels=logits_labels,
            accuracy=accuracy,
            accuracy_d=accuracy_d
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_masks: torch.Tensor,
        keep_confidence: float=0,
        min_error_prob: float=0
    ): 
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask
            )
            probability_labels = F.softmax(outputs.logits_labels, dim=-1)
            probability_d = F.softmax(outputs.logits_d, dim=-1)

            # Get actual labels considering inference parameters.
            keep_index = self.config.label2id[self.config.keep_label]
            probability_labels[:, :, keep_index] += keep_confidence
            incor_idx = self.config.d_label2id[self.config.incorrect_label]
            probability_d = probability_d[:, :, incor_idx]
            max_error_probability = torch.max(probability_d * word_masks, dim=-1)[0]
            probability_labels[max_error_probability < min_error_prob, :, keep_index] \
                                                                        = float('inf')
            pred_label_ids = torch.argmax(probability_labels, dim=-1)

            def convert_ids_to_labels(ids, id2label):
                labels = []
                for id in ids.tolist():
                    labels.append(id2label[id])
                return labels

            pred_labels = []
            for ids in pred_label_ids:
                labels = convert_ids_to_labels(
                    ids,
                    self.config.id2label
                )
                pred_labels.append(labels)

        return GECToRPredictionOutput(
            probability_labels=probability_labels,
            probability_d=probability_d,
            pred_labels=pred_labels,
            pred_label_ids=pred_label_ids,
            max_error_probability=max_error_probability
        )
    
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_pretrained(save_dir)
        torch.save(
            self.state_dict(),
            os.path.join(save_dir, 'pytorch_model.bin')
        )

    @classmethod
    def from_pretrained(self, restore_dir):
        if os.path.exists(restore_dir):
            config = GECToRConfig.from_pretrained(restore_dir)
            model = GECToR(config)
            model.load_state_dict(torch.load(
                os.path.join(restore_dir, 'pytorch_model.bin')
            ))
            return model
        else:
            card = ModelCard.load(restore_dir)
            if 'GECToR_gotutiyan' not in card.data['tags']:
                raise ValueError('Please specify the model_id that has "GECToR_gotutiyan" tag.')
            dir = snapshot_download(restore_dir, repo_type="model")
            print(dir)
            config = GECToRConfig.from_pretrained(dir)
            model = GECToR(config)
            model.load_state_dict(torch.load(
                os.path.join(dir, 'pytorch_model.bin')
            ))
            return model