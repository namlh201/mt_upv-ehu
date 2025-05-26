import os
import argparse
# from datetime import datetime
from functools import partial
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
# parser.add_argument('--model_path', type=str, required=True, help='Model path.')
parser.add_argument('--src_file', type=str, required=True, help='Reference file path.')
parser.add_argument('--can_file', type=str, required=True, help='Candidate file path.')
parser.add_argument('--ref_file', type=str, required=True, help='Reference file path.')
# parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--job_id', type=str, required=True, help='Job ID on the cluster.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
os.environ['HF_HOME'] = args.checkpoints_dir

from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import numpy as np
import torch
# from torch import nn
# from torch.utils.data import DataLoader
from transformers import logging, pipeline 
from tqdm import tqdm
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
# import wandb

load_dotenv()
logging.set_verbosity_error()

# from utils.prompt import format_to_llama3_chat
# from utils.training import train_step
# from utils.common import get_config, clean_english_data, clean_data, filter_data
# from utils.dataset import get_wikimedia_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args: argparse.Namespace, config: SimpleNamespace):
    max_seq_length = 1024

    translator = pipeline(
        task="translation",
        model="facebook/nllb-200-3.3B",
        device=device,
        src_lang="eng_Latn",
        tgt_lang="rus_Cyrl",
        max_length=max_seq_length
        # torch_dtype=torch.bfloat16
    )

    train_dataset = Dataset.from_json(os.path.join(args.data_dir, "news-commentary-v18", "news-commentary-v18.en-ru.jsonl"))

    train_dataset = train_dataset.batch(batch_size=args.batch_size)

    test_datasets = {
        "news-commentary-v18": train_dataset,
    }

    res_dir = os.path.join(args.data_dir, "res_nllb_3.3B")
    os.makedirs(res_dir, exist_ok=True)

    for name, dataset in test_datasets.items():
        src_f = open(os.path.join(res_dir, args.src_file), 'w')
        can_f = open(os.path.join(res_dir, args.can_file), 'w')
        ref_f = open(os.path.join(res_dir, args.ref_file), 'w')

        for batch in tqdm(dataset):
            srcs = batch['en']
            refs = batch['ru']

            cans = []

            # for src in srcs:
            _cans = translator(
                srcs,
                # src_lang="eng_Latn",
                # tgt_lang="rus_Cyrl"
                batch_size=args.batch_size
            )

            for can in _cans:
                can = can['translation_text']
                cans.append(can.strip())

            for src, can, ref in zip(srcs, cans, refs):
                print(src, file=src_f)
                print(can, file=can_f)
                print(ref, file=ref_f)

        src_f.close()
        can_f.close()
        ref_f.close()

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    if args.debug:
        print(args)
        # print(config)

    main(args, config=None)