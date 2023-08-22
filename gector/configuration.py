import os
import json
class GECToRConfig:
    def __init__(
        self,
        model_id: str='bert-base-cased',
        id2label: dict=None,
        label2id: dict=None,
        d_id2label: dict=None,
        d_label2id: dict=None,
        p_dropout: float=0,
        label_pad_token: str='<PAD>',
        label_oov_token: str='<OOV>',
        d_pad_token: str='<PAD>',
        keep_label: str='$KEEP',
        correct_label: str='$CORRECT',
        incorrect_label: str='$INCORRECT',
        max_length: int=128,
        label_smoothing: float=0.0,
        initializer_range: float=0.02,
        has_add_pooling_layer: bool=True,
        **kwards
    ):
        self.model_id = model_id
        self.label2id = label2id
        if id2label is None:
            self.id2label = {v:k for k,v in label2id.items()}
        else:
            self.id2label = id2label
        self.d_label2id = d_label2id
        if d_id2label is None:
            self.d_id2label = {v:k for k,v in d_label2id.items()}
        else:
            self.d_id2label = d_id2label
        self.n_labels = len(self.id2label.keys())
        self.n_d_labels = len(self.d_id2label.keys())
        self.p_dropout = p_dropout
        self.label_pad_token = label_pad_token
        self.label_oov_token = label_oov_token
        self.d_pad_token = d_pad_token
        self.keep_label = keep_label
        self.correct_label = correct_label
        self.incorrect_label = incorrect_label
        self.max_length=max_length
        self.label_smoothing=label_smoothing
        self.initializer_range = initializer_range
        self.has_add_pooling_layer = has_add_pooling_layer

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(self, restore_dir, not_dir=False):
        path = restore_dir if not_dir else os.path.join(restore_dir, 'config.json') 
        config_dict = json.load(open(path))
        config = GECToRConfig(**config_dict)
        config.id2label = {int(k):v for k,v in config.id2label.items()}
        config.d_id2label = {int(k):v for k,v in config.d_id2label.items()}
        return config

    def __repr__(self):
        return json.dumps(self.__dict__)