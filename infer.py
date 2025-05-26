import os
import sys
import argparse
# from datetime import datetime
from functools import partial
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
parser.add_argument('--model_path', type=str, required=True, help='Model path.')
parser.add_argument('--src_file', type=str, required=True, help='Reference file path.')
parser.add_argument('--can_file', type=str, required=True, help='Candidate file path.')
parser.add_argument('--ref_file', type=str, required=True, help='Reference file path.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--job_id', type=str, required=True, help='Job ID on the cluster.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
os.environ['HF_HOME'] = args.checkpoints_dir

from dotenv import load_dotenv
from datasets import load_dataset#, Dataset
import numpy as np
import torch
from transformers import logging, LlamaForCausalLM, Gemma3ForCausalLM, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

load_dotenv()
logging.set_verbosity_error()

from utils.prompt import format_to_chat
# from utils.training import train_step
from utils.common import get_config
# from utils.dataset import get_wikimedia_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args: argparse.Namespace, config: SimpleNamespace):
    if "llama" in args.model_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    elif "gemma" in args.model_path.lower():
        model = Gemma3ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "llama" in args.model_path.lower():
        if not tokenizer.pad_token:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"

    if args.debug:
        print(model)

    # EusParallel-train
    test_dataset = load_dataset(
        'google/wmt24pp',
        "en-ru_RU",
        split='train',
        cache_dir=args.data_dir
        # cache_dir=args.checkpoints_dir
    )

    # test_dataset.cleanup_cache_files()

    test_dataset = test_dataset.select(range(1, len(test_dataset)))

    test_dataset = test_dataset.map(
        lambda example: {
            "en": example["source"],
            "ru": example["target"]
        },
        batched=False,
        load_from_cache_file=False,
    )

    test_dataset = test_dataset.map(
        partial(
            format_to_chat,
            tokenizer=tokenizer,
            model="llama" if "llama" in args.model_path.lower() else "gemma"
        ),
        batched=True,
        load_from_cache_file=False,
        # remove_columns=flores_dataset.column_names
    )

    test_dataset = test_dataset.batch(batch_size=args.batch_size)

    test_datasets = {
        "WMT24": test_dataset,
    }

    res_dir = os.path.join(os.getcwd(), "res", config.model.name.split("/")[1], args.job_id)
    os.makedirs(res_dir, exist_ok=True)

    generation_config = GenerationConfig(
        num_beams=4,
        early_stopping=True,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9
    )

    for name, dataset in test_datasets.items():
        src_f = open(os.path.join(res_dir, name + '_' + args.src_file), 'w')
        can_f = open(os.path.join(res_dir, name + '_' + args.can_file), 'w')
        ref_f = open(os.path.join(res_dir, name + '_' + args.ref_file), 'w')

        for batch in tqdm(dataset):
            srcs = [en_src for en_src in batch['en']]
            refs = [ru_ref for ru_ref in batch['ru']]

            # inputs = tokenizer(batch['prompt'], padding=True, return_tensors='pt', padding_side='left').to(device)
            # print(inputs, file=sys.stderr)

            # print(batch["prompt"], file=sys.stderr)

            inputs = tokenizer(batch['prompt'], padding=True, return_tensors='pt', padding_side='left', add_special_tokens=False).to(device)
            # print(inputs, file=sys.stderr)

            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )

            cans = []

            for i in range(len(outputs)):
                output = outputs[i][len(inputs[i]):]

                can = tokenizer.decode(output, skip_special_tokens=True)

                can = ' '.join(can.split())

                cans.append(can.strip())

            for src, can, ref in zip(srcs, cans, refs):
                # print(can, file=sys.stderr)
                print(src, file=src_f)
                print(can, file=can_f)
                print(ref, file=ref_f)

        src_f.close()
        can_f.close()
        ref_f.close()

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    config_type = os.path.basename(args.config)
    config_type = os.path.splitext(config_type)[0]

    config = get_config(args.config)
    config.type = config_type

    # wandb.login(key=os.getenv('WANDB_TOKEN'))

    # # now = datetime.now()
    # # now = now.strftime('%Y%m%d_%H%M%S')
    # run = wandb.init(
    #     config=config,
    #     project=f'latxa_{config.model.name.split("/")[1]}_{config.type}',
    #     name=f'{config.model.name.split("/")[1]}_{args.job_id}'
    # )

    if args.debug:
        print(args)
        print(config)

    main(args, config)