import torch
import os


device = ("cuda" if torch.cuda.is_available() else "cpu")
root_dir = os.path.dirname(os.path.abspath(__file__))
default_bert_model = "avichr/heBERT"
global_seed=42