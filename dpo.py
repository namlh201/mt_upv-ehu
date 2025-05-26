import os
import sys
import argparse
from functools import partial
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
parser.add_argument('--grad_accum_step', default=4, type=int, help='Gradient accumulation steps.')
parser.add_argument('--model_path', type=str, required=True, help='Model path.')
parser.add_argument('--steps', default=10000, type=int, help='Number of steps.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--job_id', type=str, required=True, help='Job ID on the cluster.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
os.environ['HF_HOME'] = args.checkpoints_dir

from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel

from dotenv import load_dotenv
from datasets import load_dataset, Dataset
import numpy as np
import torch
from transformers import logging
from trl import DPOTrainer, DPOConfig
import wandb

load_dotenv()
logging.set_verbosity_error()

from utils.common import get_config
from utils.prompt import format_to_chat
from utils.training import EvalCallback

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args: argparse.Namespace, config: SimpleNamespace):
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
        # use_gradient_checkpointing=False,
    )

    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_params.r,
        target_modules=config.model.target_modules,
        lora_alpha=config.lora_params.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    if args.debug:
        model.print_trainable_parameters()

    train_dataset = Dataset.from_json(os.path.join(args.data_dir, "news-commentary-v18_pref", "news-commentary-v18.en-ru.jsonl"))
    train_dataset = train_dataset.map(
        lambda example: { 
            'prompt': [example['prompt']],
            'chosen': [example['chosen']],
            'rejected': [example['rejected']]
        }
    )

    if "llama" in config.model.name.lower():
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "gemma" in config.model.name.lower():
        response_part = "<start_of_turn>model\n"

    wmt23_dataset = Dataset.from_json(os.path.join(args.data_dir, "wmt23", "generaltest2023.en-ru.jsonl"))

    wmt23_dataset = wmt23_dataset.map(
        partial(
            format_to_chat,
            tokenizer=tokenizer,
            response_part=response_part,
            model="llama" if "llama" in args.model_path.lower() else "gemma"    
        ),
        batched=True,
        load_from_cache_file=False,
        # remove_columns=eval_dataset.column_names
    )

    print(wmt23_dataset["prompt"][0], file=sys.stderr)

    eval_datasets = {
        "WMT23": wmt23_dataset,
    }

    empty_dataset = Dataset.from_dict({
        'prompt': ['empty', 'empty'],
        'chosen': ['empty', 'empty'],
        'rejected': ['empty', 'empty']
    })

    num_train_epochs = 1

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=empty_dataset, 
        args=DPOConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_step,
            beta=config.train_params.beta,
            loss_type=config.train_params.loss_type,
            warmup_steps=100,
            num_train_epochs=num_train_epochs, # Set this for 1 full training run.
            # max_steps=args.steps,
            save_steps=250,
            learning_rate=config.train_params.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            optim='adamw_8bit',
            weight_decay=config.train_params.weight_decay,
            lr_scheduler_type='cosine',
            seed=3407,
            output_dir=f'models/{config.model.name.split("/")[1]}-DPO/{args.job_id}',
            report_to='none', # Use this for WandB etc
            disable_tqdm=False,
            eval_strategy='steps',
            eval_steps=250,
            max_length=max_seq_length,
            dataset_num_proc=2,
        )
    )

    eval_callback = EvalCallback(trainer, eval_datasets, is_unsloth=True, use_gradient_checkpointing=True)
    trainer.add_callback(eval_callback)

    trainer_stats = trainer.train()

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    config_type = os.path.basename(args.config)
    config_type = os.path.splitext(config_type)[0]

    config_type = 'd' + config_type[1:]

    config = get_config(args.config)
    config.type = config_type

    wandb.login(key=os.getenv('WANDB_TOKEN'))

    run = wandb.init(
        config=config,
        project=f'mt_proj_{config.model.name.split("/")[1]}',
        name=f'dpo_{args.job_id}',
        # id=args.wandb_run_id if args.resume and args.wandb_run_id else None,
        # resume='allow'
    )

    if args.debug:
        print(args)
        print(config)

    main(args, config)

    wandb.finish()