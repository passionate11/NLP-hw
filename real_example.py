# Load spacy tokenizer models, download them if they haven't been
# downloaded already
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import subsequent_mask, make_model, execute_example, DummyOptimizer, DummyScheduler
import sys
sys.path.append('/extra/changkaiyan/hpc/nlp-course/final_hw/annotated_transformer')
from model_training import train_model

from utils import load_tokenizers, load_vocab
   
# if is_interactive_notebook():
#     # global variables used later in the script
#     spacy_de, spacy_en = show_example(load_tokenizers)
#     vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

def load_trained_model(file_prefix_dir):
    config = {
        "batch_size": 128,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        'file_prefix': file_prefix_dir,
        #"file_prefix": "/extra/changkaiyan/hpc/nlp-course/final_hw/annotated_transformer/model/multi30k_model_",
    }
    # model_path = "/extra/changkaiyan/hpc/nlp-course/final_hw/annotated_transformer/model/multi30k_model_final.pt"
    model_path = file_prefix_dir + 'final.pt'
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path))
    return model

def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    file_prefix_dir = '/extra/changkaiyan/hpc/nlp-course/final_hw/annotated_transformer/model/12_8/model_'
    model = load_trained_model(file_prefix_dir)