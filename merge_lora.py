import os
import argparse
from functools import partial
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Model path.')
parser.add_argument('--model_type', type=str, required=True, choices=['base', 'instruct'], help='Model type (`base` or `instruct`).')
parser.add_argument('--save_path', type=str, required=True, help='Merged model save path.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--job_id', type=str, required=True, help='Job ID on the cluster.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')
parser.add_argument('--unsloth', default=True, action='store_true', help='Merge with Unsloth.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
os.environ['HF_HOME'] = args.checkpoints_dir

from unsloth import FastLanguageModel

from dotenv import load_dotenv
import numpy as np
import torch
from transformers import logging, Gemma3ForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, PeftModel, AutoPeftModelForCausalLM

# import wandb

load_dotenv()
logging.set_verbosity_error()

from utils.common import get_config

from tensorflow.compat.v1 import logging as tf_logging
tf_logging.set_verbosity(tf_logging.ERROR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main_unsloth(args: argparse.Namespace, config: SimpleNamespace):
    max_seq_length = 4096

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=config.lora.load_in_4bit,
        load_in_8bit=config.lora.load_in_8bit,
        device_map='sequential',
        token=os.environ['HF_TOKEN'],
        cache_dir=args.checkpoints_dir,
        use_exact_model_name=True,
        use_gradient_checkpointing=False,
    )

    # if hasattr(model, "language_model"):
    #     model = model.language_model
        # model.max_seq_length = max_seq_length

    model.save_pretrained_merged(args.save_path, tokenizer, save_method='merged_16bit')
    
def main_tf(args: argparse.Namespace, config: SimpleNamespace):
    # model = Gemma3ForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     attn_implementation="flash_attention_2"
    # )
    # print(model)
    model = AutoPeftModelForCausalLM.from_pretrained(
        # model,
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(model)
    model.merge_and_unload()
    print(model)

    # tokenizer.save_pretrained(args.save_path)
    # model.save_pretrained(args.save_path)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    config_type = os.path.basename(args.config)
    config_type = os.path.splitext(config_type)[0]

    config = get_config(args.config)
    config.type = config_type

    if args.debug:
        print(args)
        print(config)

    if args.unsloth:
        main_unsloth(args, config)
    else:
        main_tf(args, config)

    # wandb.finish()