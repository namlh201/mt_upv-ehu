import os
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
from datasets import Dataset
import numpy as np
import torch
from transformers import logging, PreTrainedTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

load_dotenv()
logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def format_prompt(example, tokenizer: PreTrainedTokenizer) -> dict[str, list[str]]:
    en_src = example["en"]
    ru_ref = example["ru"]

    text_convo = [
        {
            "role": "system",
            "content": "You are a translation assistant specifically designed to provide accurate and contextually appropriate translations. Please ensure that your translation accurately captures the meaning and nuances of the English text, while also taking into account any cultural or linguistic differences that may apply."
        },
        {
            "role": "user",
            "content": f"Translate the following text from English to Russian.\nEnglish: {en_src}\nRussian: "
        },
        {
            "role": "assistant",
            "content": ru_ref
        }
    ]
    prompt_convo = [
        {
            "role": "system",
            "content": "You are a translation assistant specifically designed to provide accurate and contextually appropriate translations. Please ensure that your translation accurately captures the meaning and nuances of the English text, while also taking into account any cultural or linguistic differences that may apply."
        },
        {
            "role": "user",
            "content": f"Translate the following text from English to Russian.\nEnglish: {en_src}\nRussian: "
        }
    ]

    text = tokenizer.apply_chat_template(text_convo, tokenize=False)
    prompt = tokenizer.apply_chat_template(prompt_convo, tokenize=False, add_generation_prompt=True)

    return {
        "text": text,
        "prompt": prompt
    }

def format_prompt_batch(examples, tokenizer: PreTrainedTokenizer) -> dict[str, list[str]]:
    texts = []
    prompts = []

    en_src_list = examples["en"]
    ru_ref_list = examples["ru"]

    for en_src, ru_ref in zip(en_src_list, ru_ref_list):
        text_convo = [
            {
                "role": "system",
                "content": "You are a translation assistant specifically designed to provide accurate and contextually appropriate translations. Please ensure that your translation accurately captures the meaning and nuances of the English text, while also taking into account any cultural or linguistic differences that may apply."
            },
            {
                "role": "user",
                "content": f"Translate the following text from English to Russian.\nEnglish: {en_src}\nRussian: "
            },
            {
                "role": "assistant",
                "content": ru_ref
            }
        ]
        prompt_convo = [
            {
                "role": "system",
                "content": "You are a translation assistant specifically designed to provide accurate and contextually appropriate translations. Please ensure that your translation accurately captures the meaning and nuances of the English text, while also taking into account any cultural or linguistic differences that may apply."
            },
            {
                "role": "user",
                "content": f"Translate the following text from English to Russian.\nEnglish: {en_src}\nRussian: "
            }
        ]

        text = tokenizer.apply_chat_template(text_convo, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt_convo, tokenize=False, add_generation_prompt=True)

        texts.append(text)
        prompts.append(prompt)

    return {
        "text": texts,
        "prompt": prompts
    }
    

def main(args: argparse.Namespace, config: SimpleNamespace):
    max_seq_length = 1024

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        max_model_len=max_seq_length,
        gpu_memory_utilization=0.9
    )

    tokenizer = llm.get_tokenizer()

    train_dataset = Dataset.from_json(os.path.join(args.data_dir, "news-commentary-v18", "news-commentary-v18.en-ru.jsonl"))

    train_dataset = train_dataset.map(
        partial(format_prompt_batch, tokenizer=tokenizer),
        batched=True,
        load_from_cache_file=False,
        # remove_columns=flores_dataset.column_names
    )

    train_dataset = train_dataset.batch(len(train_dataset))

    test_datasets = {
        "news-commentary-v18": train_dataset,
    }

    res_dir = os.path.join(args.data_dir, f"res_{args.model_path.split('/')[-1]}")
    os.makedirs(res_dir, exist_ok=True)

    sampling_params = SamplingParams(
        max_tokens=max_seq_length,
        top_p=0.9,
        top_k=50,
    )

    for name, dataset in test_datasets.items():
        src_f = open(os.path.join(res_dir, args.src_file), 'w')
        can_f = open(os.path.join(res_dir, args.can_file), 'w')
        ref_f = open(os.path.join(res_dir, args.ref_file), 'w')

        for batch in tqdm(dataset):
            srcs = batch['en']
            refs = batch['ru']

            outputs = llm.generate(
                batch['prompt'],
                sampling_params=sampling_params
            )

            cans = [' '.join(output.outputs[0].text.split()).strip() for output in outputs]

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

    config_type = os.path.basename(args.config)
    config_type = os.path.splitext(config_type)[0]

    if args.debug:
        print(args)
        # print(config)

    main(args, config=None)