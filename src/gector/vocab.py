from .configuration import GECToRConfig
from .dataset import GECToRDataset
import os

def build_vocab(
    train_dataset: GECToRDataset,
    n_max_labels: int=5000,
    n_max_d_labels: int=2
):
    label2id = {'<OOV>':0, '$KEEP':1}
    d_label2id = {'$CORRECT':0, '$INCORRECT':1, '<PAD>':2}
    freq_labels, _ = train_dataset.get_labels_freq(
        exluded_labels=['<PAD>'] + list(label2id.keys())
    )

    def get_high_freq(freq: dict, n_max: int):
        descending_freq = sorted(
            freq.items(), key=lambda x:x[1], reverse=True
        )
        high_freq = [x[0] for x in descending_freq][:n_max]
        if len(high_freq) < n_max:
            print(f'Warning: the size of the vocablary: {len(high_freq)} is less than n_max: {n_max}.')
        return high_freq
    
    high_freq_labels = get_high_freq(freq_labels, n_max_labels-2)
    for i, x in enumerate(high_freq_labels):
        label2id[x] = i + 2
    label2id['<PAD>'] = len(label2id)
    return label2id, d_label2id

def load_vocab_from_config(config_file: str):
    config = GECToRConfig.from_pretrained(config_file, not_dir=True)
    return config.label2id, config.d_label2id

def load_vocab_from_official(dir):
    vocab_path = os.path.join(dir, 'labels.txt')
    vocab = open(vocab_path).read().replace('@@PADDING@@', '').replace('@@UNKNOWN@@', '').rstrip().split('\n')
    # vocab_d = open(dir + 'd_tags.txt').read().rstrip().replace('@@PADDING@@', '<PAD>').replace('@@UNKNOWN@@', '<OOV>').split('\n')
    label2id = {'<OOV>':0, '$KEEP':1}
    d_label2id = {'$CORRECT':0, '$INCORRECT':1, '<PAD>':2}
    idx = len(label2id)
    for v in vocab:
        if v not in label2id:
            label2id[v] = idx
            idx += 1
    label2id['<PAD>'] = idx
    return label2id, d_label2id
        