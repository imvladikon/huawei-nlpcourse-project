import argparse
from torchsummary import summary

import const
from processing.gru_ae import GRUAE
from utils import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(**kwargs):
    print_gpu()
    device = prepare_device(kwargs["no_cuda"])
    print(device)
    input_file = kwargs["input_file"]
    output_file = kwargs["output_file"]
    config_file = kwargs["config"]
    state_dict = kwargs["state_dict"]
    config = json.load(open(config_file))
    model = GRUAE(**config['model']['args'])
    state_dict = os.path.abspath(state_dict)
    if os.path.exists(state_dict):
        checkpoint = torch.load(state_dict)
        model.load_state_dict(checkpoint["state_dict"])
    # print(model)
    model.to(device).eval()
    summary(model, (1, 768))
    data_info = json.load(open(input_file))
    embedding_file = data_info['embedding_file']
    if os.path.relpath(embedding_file):
        embedding_file = os.path.join(os.path.abspath(os.path.dirname(input_file)), embedding_file)
    data = load_data(embedding_file)
    shape = list(map(int, data_info["shape"]))
    data_size = len(data)
    result_tensor = torch.zeros((data_size, 512), dtype=torch.float32)
    # if we read sequentially embeddings file, mapping file should stay the same
    with torch.no_grad():
        for idx, datum in tqdm(enumerate(data), total=data_size):
            out_vector = model.encoder(torch.from_numpy(datum).view(*shape).to(device))
            # (1,1,512) -> (512)
            result_tensor[idx] = out_vector.view(512).cpu()
    save_data(result_tensor, output_file)
    print("features saved to the {}".format(os.path.abspath(output_file)))
    data_info["features_extracted_file"] = os.path.relpath(output_file, start=os.path.dirname(input_file))
    json.dump(data_info, open(input_file, "w"))
    del data
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction by GRU-AE')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument("-i", "--input_file", default=None, type=str, required=True)
    parser.add_argument("-o", "--output_file", default=None, type=str, required=True, help="output file")
    parser.add_argument("-s", "--state_dict", default=None, type=str, required=True,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument("--no-cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=const.global_seed,
                        help="random seed for initialization")

    args = parser.parse_args()
    print("setup seed: {}".format(args.seed))
    setup_seed(args.seed)
    main(**vars(args))