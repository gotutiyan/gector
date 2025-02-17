import os
import json
from transformers import PretrainedConfig
class GECToRConfig(PretrainedConfig):
    def __init__(
        self,
        model_id: str = 'bert-base-cased',
        p_dropout: float=0,
        label_pad_token: str='<PAD>',
        label_oov_token: str='<OOV>',
        d_pad_token: str='<PAD>',
        keep_label: str='$KEEP',
        correct_label: str='$CORRECT',
        incorrect_label: str='$INCORRECT',
        label_smoothing: float=0.0,
        has_add_pooling_layer: bool=True,
        initializer_range: float=0.02,
        is_official_model: bool=False,
        **kwards
    ):
        super().__init__(**kwards)
        self.d_label2id = {
            "$CORRECT": 0,
            "$INCORRECT": 1,
            "<PAD>": 2
        }
        self.d_id2label = {v: k for k, v in self.d_label2id.items()}
        self.d_num_labels = len(self.d_label2id)
        self.model_id = model_id
        self.p_dropout = p_dropout
        self.label_pad_token = label_pad_token
        self.label_oov_token = label_oov_token
        self.d_pad_token = d_pad_token
        self.keep_label = keep_label
        self.correct_label = correct_label
        self.incorrect_label = incorrect_label
        self.label_smoothing = label_smoothing
        self.has_add_pooling_layer = has_add_pooling_layer
        self.initializer_range = initializer_range
        self.is_official_model = is_official_model
