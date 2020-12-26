import argparse
import gc
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
import tqdm as tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertTokenizer, BertTokenizerFast
import os
import pandas as pd
from tqdm import tqdm
import regex as re
from utils import *
import json
import shutil
import const

try:
    import en_core_sci_sm as en

    nlp = en.load()


    def spacy_splitter(s):
        return [t.text for t in nlp(s)]

except:
    pass


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from const import default_bert_model
from preprocessing.string_utils import strip_accents


class HebrewNewsDataset(Dataset):

    def __init__(self, data, preprocessing=None, max_length=512, tokenizer: PreTrainedTokenizer = None):
        self.data = data
        self.preprocessing = preprocessing
        self.max_length = max_length
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(default_bert_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx][0]
        if self.preprocessing is not None:
            record = type(record)(map(self.preprocessing, record))
        record = self.tokenizer.encode_plus(record,
                                            return_tensors="pt",
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            max_length=self.max_length,
                                            pad_to_max_length=True,
                                            is_pretokenized=False,
                                            truncation=True)
        return {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "token_type_ids": record["token_type_ids"]
        }



    @staticmethod
    def calc_max_seq_len(in_data, tokenizer):
        if isinstance(in_data, str):
            in_data = [in_data]
        print("calculating max_length")
        max_length = 0
        for t in tqdm(in_data):
            m = len(tokenizer.tokenize(t))
            if m > max_length:
                max_length = m
        return max_length

    @staticmethod
    def from_data(data, tokenizer, max_length, batch_size=1):
        ds = HebrewNewsDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=1
        )



def pooling_last4(t: torch.Tensor):
    return F.softmax(torch.sum(t[-4:], dim=0), dim=0)


def pooling_mean(t: torch.Tensor):
    return t.mean(dim=0)


class BertEmbeddingLayer(torch.nn.Module):

    def __init__(self, model: BertModel,
                 transform=None,
                 device=const.device):
        super(BertEmbeddingLayer, self).__init__()
        self.model = model
        self.transform = transform
        self.device = device

    def to(self, *args, **kwargs):
        if "device" in kwargs:
            self.device = kwargs["device"]
        elif args:
            self.device = args[0]
        return self

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # last_hidden_state
            _, pooled_output = self.model(input_ids, attention_mask=attention_mask)
        if self.transform:
            pooled_output = self.transform(pooled_output)
        return pooled_output






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating Tokens and Embeddings from Text')
    parser.add_argument("-i", "--input_file", default=None, type=str, required=True)
    parser.add_argument("--column_name", default="articleBody", type=str, required=True,
                        help="specify text column, e.g. 'articleBody'")
    parser.add_argument("-o", "--output_dir", default=None, type=str, required=True)
    parser.add_argument("-f", "--format_file", default="pt", type=str, required=True)
    parser.add_argument("-b", "--bert_model", default=None, type=str,
                        help="path or name of the BERT model")
    parser.add_argument("-p", "--pretokenized", action='store_true', help="use spacy for pretokenizing")
    parser.add_argument("--no-cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length by 512 chunks (max_len=max_seq_length*512). Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded. If None - auto calc")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for embeddings")
    parser.add_argument("--save_tokens", action='store_true', default=False, help="do save tokens and attention masks")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    print("setup seed: {}".format(args.seed))
    setup_seed(args.seed)

    in_data = pd.read_csv(args.input_file)
    assert isinstance(in_data, pd.DataFrame), "please, provide pandas data frame for input data"
    in_data = in_data.reset_index(drop=True)
    in_data[args.column_name] = in_data[args.column_name].fillna("").values

    outfile_ext = args.format_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print_gpu()
    device = prepare_device(args.no_cuda)
    print(device)

    bert_path = args.bert_model if args.bert_model is not None else const.default_bert_model
    model = BertModel.from_pretrained(bert_path).to(device).eval()
    max_position_embeddings = model.config.max_position_embeddings
    max_hidden_size = model.config.hidden_size

    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=args.do_lower_case)
    if args.max_seq_length is None or args.max_seq_length == 0:
        max_seq_len_text = HebrewNewsDataset.calc_max_seq_len(in_data=in_data[args.column_name].values,
                                                              tokenizer=BertTokenizerFast.from_pretrained(
                                                                  bert_path)) // max_position_embeddings + 1
    else:
        max_seq_len_text = args.max_seq_length
    print("max len of the tokens {} is {} ".format(args.column_name, max_seq_len_text))
    max_len = max_position_embeddings * max_seq_len_text
    # embedding_layer = BertEmbeddingLayer(model, transform=torch.nn.Dropout(p=0.3)).to(device)
    embedding_layer = BertEmbeddingLayer(model).to(device)
    data_loader = HebrewNewsDataset.from_data(in_data[[args.column_name]].values, tokenizer,
                                              max_length=max_len, batch_size=args.batch_size)
    data_size = len(in_data)
    cpu_device = torch.device("cpu")
    t_input_ids = torch.zeros((data_size, 1, max_seq_len_text * max_position_embeddings), dtype=torch.long).to(
        cpu_device) if args.save_tokens else None
    t_attention_mask = torch.zeros((data_size, 1, max_seq_len_text * max_position_embeddings), dtype=torch.long).to(
        cpu_device) if args.save_tokens else None
    t_pooled_output = torch.zeros((data_size, 1, max_seq_len_text * max_hidden_size), dtype=torch.float32).to(
        cpu_device)
    idx = 0
    new_col_index = "index_embeddings_{}".format(args.column_name.lower())
    for d in tqdm(data_loader, total=data_size // args.batch_size + int(data_size % args.batch_size > 0)):
        batch_size = len(d["input_ids"])
        batch_input_ids = d["input_ids"].to(cpu_device)
        batch_attention_mask = d["attention_mask"].to(cpu_device)
        if args.save_tokens:
            t_input_ids[idx: idx + batch_size] = batch_input_ids
            t_attention_mask[idx: idx + batch_size] = batch_attention_mask
        for j in range(batch_size):
            input_ids = batch_input_ids[j].view(max_seq_len_text, max_position_embeddings).to(device)
            attention_mask = batch_attention_mask[j].view(max_seq_len_text, max_position_embeddings).to(device)
            t_pooled_output[idx + j] = embedding_layer(input_ids, attention_mask).flatten().detach().cpu()
        idx += batch_size
        gc.collect()
        torch.cuda.empty_cache()
    output_file = os.path.join(output_dir, "embeddings_{}.{}".format(args.column_name.lower(), outfile_ext))
    save_data(t_pooled_output.detach().cpu(), output_file)
    if args.save_tokens:
        torch.save(t_input_ids.detach().cpu(), os.path.join(output_dir, "input_ids.pt"))
        torch.save(t_attention_mask.detach().cpu(),
                   os.path.join(output_dir, "attention_mask.pt"))
    print("copying input data to the embeddings folder")
    shutil.copy(args.input_file, output_dir)
    info_data = dict(column=args.column_name,
                     embedding_file=os.path.relpath(output_file, output_dir),
                     embedding_file_url="",
                     len=len(t_pooled_output),
                     shape=[max_seq_len_text, max_hidden_size],
                     data_file=os.path.relpath(os.path.basename(args.input_file), output_dir),
                     features_extracted_file="")
    json.dump(info_data, open(os.path.join(output_dir, "data_info.json"), "w"), indent=4, sort_keys=True)
    try:
        del t_pooled_output, t_input_ids, t_attention_mask
    except:
        pass
    torch.cuda.empty_cache()
    gc.collect()
