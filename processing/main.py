import json

import torch
from utils import from_pickle, to_hickle, from_h5
from processing.gru_ae import GRUAE
import const

def main():
    config = json.load(open('../config/article_body.json'))
    model = GRUAE(**config['model']['args']).to(const.device)
    dict_state = torch.load("saved/articleBody/1226_160301/checkpoint-epoch30.pth")["state_dict"]
    model.load_state_dict(dict_state)
    # print(model)
    data = from_h5("../preprocessing/embeddings/embeddings_articlebody.h5")
    mse = torch.nn.MSELoss()

    result = []
    for i in range(len(data)):
        in_vector = torch.from_numpy(data[i]).to(const.device)  # .mean(dim=1)
        out_vector = model.encoder(in_vector)
        result.append(out_vector.detach().cpu().numpy())
    to_hickle(in_object=result, filename="/root/G-Bert/data/discharges_embeddings_output_cpu.hkl")

    # out_vector = model(in_vector)

    # ms = mse(in_vector, out_vector)
    # i = 42

    # summ_args = summ_config['model']['args']
    # self.summarizer = GRUAE(**summ_args)
    # self.summarizer.load_state_dict(state_dict)
    # with torch.no_grad():
    #     hidden = self.summarizer.encoder.init_hidden(batch_size, device)
    #     enc = self.summarizer.encoder(x_text, hidden, x_tl, b_is)  # sentence summarizer
    #     enc = enc.squeeze()


if __name__ == '__main__':
    main()
