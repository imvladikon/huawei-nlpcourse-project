from collections import defaultdict
from typing import List, Union

from tqdm import tqdm
import numpy as np

from utils.string_utils import preprocessing
from tokenizer import HebTokenizer


def preprocess_text(texts:Union[np.ndarray, List])-> List:
    heb_tokenizer = HebTokenizer()

    result = []
    for line in tqdm(texts, desc="text preprocessing"):
        tokens = heb_tokenizer.get_words(preprocessing(line.lower()))
        if len(tokens) == 0:
            continue
        result.append(tokens)

    return result

class Vocab():

    def __init__(self, texts, max_size:int=50_000, *args, **kwargs):
        self.sorted_word_counts = None
        self.word2freq = None
        self.max_size = max_size

        word_counts = defaultdict(int)
        total_words = 0
        docs = 0
        for txt in tqdm(texts, desc="creating vocab"):
            docs += 1
            for word in set(txt):
                word_counts[word] += 1
                total_words += 1

        self.sorted_word_counts = sorted(word_counts.items(), reverse=True, key=lambda pair: pair[1])
        self.sorted_word_counts = [('<pad>', 0), ('<unk>', 0)] + self.sorted_word_counts

        if len(word_counts) > self.max_size:
            self.sorted_word_counts = self.sorted_word_counts[:self.max_size]

        self.w2id = {word: i for i, (word, _) in enumerate(self.sorted_word_counts)}
        self.id2w = {i: word for i, (word, _) in enumerate(self.sorted_word_counts)}

        self.word2freq = np.array([cnt / docs for _, cnt in self.sorted_word_counts])

    def save(self, vocab_file:str) -> "Vocab":
       with open(vocab_file, mode='w', encoding='utf8') as vocab_file:
           for word, _ in tqdm(self.sorted_word_counts, desc="to file"):
               vocab_file.write(word + '\n')
       return self

    def texts_to_ids(self, texts: List) -> List:
        return [[self.w2id.get(token, 1) for token in text] for text in texts]


if __name__ == '__main__':
    texts = preprocess_text("../data/news_sample.csv", "articleBody")
    vocab = Vocab(texts).save("vocab.vocab")

    i = 42