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
        record = self.data[idx]
        if self.preprocessing is not None:
            record = type(record)(map(self.preprocessing, record))
        record = self.tokenizer.encode_plus(record, return_tensors="pt",
                                            return_token_type_ids=True,
                                            return_attention_mask=True, max_length=self.max_length, truncation=True)
        return {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "token_type_ids": record["token_type_ids"]
        }


def main():
    import pandas as pd
    data = pd.read_csv("../data/news_sample.csv")["articleBody"].tolist()
    ds = HebrewNewsDataset(data=data, preprocessing=strip_accents)
    train_data = DataLoader(ds, batch_size=1,)
    for record in train_data:
        print(record["input_ids"])


if __name__ == '__main__':
    main()
