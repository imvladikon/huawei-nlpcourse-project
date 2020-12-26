import json

import torch
from utils import from_pickle, to_hickle
from processing.gru_ae import GRUAE


def main():

    config = json.load(open('/root/G-Bert/processing/config/pubmed_discharge.json'))
    model = GRUAE(**config['model']['args']).to("cuda")
    print(model)
    data = from_pickle("/root/G-Bert/data/discharges_embeddings_cpu.pkl")
    mse = torch.nn.MSELoss()
    result = []
    for i in range(len(data)):
        in_vector = torch.from_numpy(data[i]).to("cuda")  # .mean(dim=1)
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
