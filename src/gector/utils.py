import inspect
from transformers import AutoModel

def has_args_add_pooling(model_id):
    model = AutoModel.from_pretrained(model_id)
    return 'add_pooling_layer' in inspect.signature(model.__init__).parameters.keys()
