from .mlp_classifier import MLPClassifier
from .yolo_v5 import YoloV5
from .T5 import T5
from .TinyLlama import TinyLlama
from .BertSquad import BertSquad

TASK_REGISTRY = {
    "yolo_v5": YoloV5,
    "T5": T5,
    "yolo_v7": None,
    "yolo_v8": None,
    "BERT": BertSquad,
    "mlp": MLPClassifier,
    "mlp_classifier": MLPClassifier,
    "tiny_llama":TinyLlama,
}