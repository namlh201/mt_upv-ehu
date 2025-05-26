import os
import argparse
from functools import partial
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
parser.add_argument('--grad_accum_step', default=4, type=int, help='Gradient accumulation steps.')
parser.add_argument('--steps', default=10000, type=int, help='Number of steps.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--job_id', type=str, required=True, help='Job ID on the cluster.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')
parser.add_argument('--resume', default=False, action='store_true', help='Resume training from the last checkpoint.')
parser.add_argument('--wandb_run_id', default=None, type=str, help='WandB run ID (for resuming training).')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
os.environ['HF_HOME'] = args.checkpoints_dir

from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel

from dotenv import load_dotenv
from datasets import Dataset
import numpy as np
import torch
from transformers import logging
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
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
        model_name=config.model.name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=config.lora.load_in_4bit,
        load_in_8bit=config.lora.load_in_8bit,
        device_map='auto',
        token=os.environ['HF_TOKEN'],
        cache_dir=args.checkpoints_dir,
        use_exact_model_name=True,
        use_gradient_checkpointing=False,
        # attn_implementation="flash_attention_2",
    )

    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    if args.debug:
        print(model.config)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_params.r,
        target_modules=config.model.target_modules,
        lora_alpha=config.lora_params.alpha,
        lora_dropout=0,
        bias="none",
        #use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
        use_gradient_checkpointing=False,
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    if args.debug:
        print(model)
        model.print_trainable_parameters()

    train_dataset = Dataset.from_json(os.path.join(args.data_dir, "news-commentary-v18_sft", "news-commentary-v18.en-ru.jsonl"))

    train_dataset = train_dataset.map(
        partial(format_to_chat, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
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
            model="llama" if "llama" in config.model.name.lower() else "gemma"  
        ),
        batched=True,
        load_from_cache_file=False,
        # remove_columns=eval_dataset.column_names
    )

    eval_datasets = {
        "WMT23": wmt23_dataset,
    }

    empty_dataset = Dataset.from_dict({
        'text': ['empty', 'empty']
    })

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_part,
        tokenizer=tokenizer
    )

    num_train_epochs = 1

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=empty_dataset,
        data_collator=data_collator,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_step,
            warmup_steps=100,
            num_train_epochs=num_train_epochs,
            # max_steps=args.steps,
            learning_rate=config.train_params.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            optim='adamw_8bit',
            weight_decay=config.train_params.weight_decay,
            lr_scheduler_type='cosine',
            seed=3407,
            output_dir=f'models/{config.model.name.split("/")[1]}/{args.job_id}',
            report_to='none',
            disable_tqdm=False,
            eval_strategy='steps',
            eval_steps=250,
            save_steps=250,
            dataset_text_field='text',
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            eval_packing=False,
        ),
    )

    eval_callback = EvalCallback(trainer, eval_datasets, is_unsloth=True)
    trainer.add_callback(eval_callback)

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    config_type = os.path.basename(args.config)
    config_type = os.path.splitext(config_type)[0]

    config = get_config(args.config)
    config.type = config_type

    wandb.login(key=os.getenv('WANDB_TOKEN'))

    run = wandb.init(
        config=config,
        project=f'mt_proj_{config.model.name.split("/")[1]}',
        name=f'sft_{args.job_id}' if not (args.resume and args.wandb_run_id) else None,
        id=args.wandb_run_id if args.resume and args.wandb_run_id else None,
        resume='allow'
    )

    if args.debug:
        print(args)
        print(config)

    main(args, config)

  
