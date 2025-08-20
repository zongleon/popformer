import torch
from models import HapbertaForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset

samples = np.load()

tokenizer = AutoTokenizer.from_pretrained("./hapberta")
model = HapbertaForSequenceClassification.from_pretrained("./hapberta")
