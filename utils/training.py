from torch import tensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from transformers import Trainer
from transformers.integrations import WandbCallback
from tqdm import tqdm

from eval import check_valid_for_metrics, format_for_metrics, METRIC_MAP

class EvalCallback(WandbCallback):
    def __init__(
        self,
        trainer: Trainer,
        eval_datasets: dict[str, Dataset],
        eval_batch_size: int=64,
        is_unsloth: bool=False,
        calculate_loss: bool=True,
        use_gradient_checkpointing: bool=False,
        wandb=None
    ):
        super().__init__()
        self.eval_datasets = eval_datasets
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.batch_size = eval_batch_size
        self.is_unsloth = is_unsloth
        self.calculate_loss = calculate_loss
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if wandb:
            self._wandb = wandb

        if is_unsloth:
            from unsloth import FastLanguageModel

            self.for_inference = FastLanguageModel.for_inference
            self.for_training = FastLanguageModel.for_training

    def loss(self, texts: list[str], label_masks: list[Tensor]=None) -> float:
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)

        if not label_masks:
            labels = inputs['input_ids'] * inputs['attention_mask'] + (-100) * (1 - inputs['attention_mask'])
        else:
            label_masks = pad_sequence(label_masks, batch_first=True).to(self.model.device)

            labels = inputs['input_ids'] * label_masks + (-100) * (1 - label_masks)

        outputs = self.model(**inputs, labels=labels)

        return outputs.loss.item()

    def generate(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            padding_side='left',
            add_special_tokens=False
        ).to(self.model.device)

        if self.is_unsloth:
            self.for_inference(self.model)
        else:
            self.model.eval()
        outputs = self.model.generate(**inputs, max_new_tokens=512, use_cache=True)
        if self.is_unsloth:
            self.for_training(self.model, use_gradient_checkpointing=self.use_gradient_checkpointing)
        else:
            self.model.train()

        generated = []
        for i in range(inputs['input_ids'].shape[0]):
            can = self.tokenizer.decode(outputs[i][len(inputs[i]):], skip_special_tokens=True)

            can = ' '.join(can.split())

            generated.append(can.strip())

        return generated

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        for name, eval_dataset in self.eval_datasets.items():
            eval_loss = 0.0

            sources = []
            references = []
            candidates = []

            if "en" in eval_dataset.column_names:
                en_src_list = eval_dataset["en"]
            elif "eng" in eval_dataset.column_names:
                en_src_list = eval_dataset["eng"]
            elif "en_src" in eval_dataset.column_names:
                en_src_list = eval_dataset["en_src"]

            if "ru" in eval_dataset.column_names:
                ru_trans_list = eval_dataset["ru"]
            elif "rus" in eval_dataset.column_names:
                ru_trans_list = eval_dataset["rus"]
            elif "ru_ref" in eval_dataset.column_names:
                ru_trans_list = eval_dataset["ru_ref"]
            
            prompt_list = eval_dataset['prompt']
            text_list = eval_dataset['text']
            label_mask_list = eval_dataset['label_mask'] if 'label_mask' in eval_dataset.column_names else None

            for i in tqdm(range(0, len(text_list), self.batch_size), desc=f'Predicting from {name} dataset'):
                en_src = en_src_list[i:i + self.batch_size]
                ru_trans = ru_trans_list[i:i + self.batch_size]
                prompts = prompt_list[i:i + self.batch_size]
                texts = text_list[i:i + self.batch_size]
                label_masks = label_mask_list[i:i + self.batch_size] if label_mask_list else None
                label_masks = list(map(tensor, label_masks)) if label_masks else None

                en_src = [' '.join(_en_src.split('\n\n')).strip() for _en_src in en_src]
                ru_trans = [' '.join(_ru_trans.split('\n\n')).strip() for _ru_trans in ru_trans]

                sources.extend(en_src)
                references.extend(ru_trans)
                candidates.extend(self.generate(prompts))

                if self.calculate_loss:
                    for bs in range(0, len(texts), 1):
                        eval_loss += self.loss(texts[bs:bs + 1], label_masks[bs:bs + 1])

            eval_res = {}

            if self.calculate_loss:
                eval_loss = eval_loss / len(eval_dataset)# * self.batch_size

                eval_res[f'eval/{name}/eval_loss'] = eval_loss

            metrics = METRIC_MAP.keys()

            for metric in metrics:
                check_valid_for_metrics(metric, sources, references, candidates)

                data = format_for_metrics(metric, sources, references, candidates)

                # score = getattr(eval, metric)(*data)
                score = METRIC_MAP[metric](*data)

                eval_res[f'eval/{name}/{metric}'] = score

            self._wandb.log(eval_res)
