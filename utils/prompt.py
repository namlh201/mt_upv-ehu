from transformers import PreTrainedTokenizer

def format_example_to_conversation_llama(en_src: str, ru_trans: str, train: bool=True) -> list[dict[str, str]]:
    user_input = {
        "role": "user",
        "content": f"Translate the following text from English to Russian.\nEnglish: {en_src}\nRussian: "
    }

    assistant_response = {
        "role": "assistant",
        "content": ru_trans
    }

    if train:
        conversation = [user_input, assistant_response]
    else:
        conversation = [user_input]
    
    return conversation

def format_example_to_conversation_gemma(en_src: str, ru_trans: str, train: bool=True) -> list[dict[str, str]]:
    user_input = {
        "role": "user",
        "content": f"Translate the following text from English to Russian. Do not give any explanation, only return the translated Russian text.\nEnglish: {en_src}\nRussian: "
    }

    assistant_response = {
        "role": "assistant",
        "content": ru_trans
    }

    if train:
        conversation = [user_input, assistant_response]
    else:
        conversation = [user_input]
    
    return conversation

def format_to_chat(
    examples,
    tokenizer: PreTrainedTokenizer,
    response_part: str=None,
    model: str="llama"
) -> dict[str, list[str]]:
    en_src_list = examples['en'] if 'en' in examples else examples['eng']
    ru_trans_list = examples['ru'] if 'ru' in examples else examples['rus']

    texts = []
    prompts = []

    if response_part:
        label_masks = []

    for en_src, ru_trans in zip(en_src_list, ru_trans_list):
        if model == "llama":
            text = format_example_to_conversation_llama(en_src, ru_trans, train=True)
            prompt = format_example_to_conversation_llama(en_src, ru_trans, train=False)
        elif model == "gemma":
            text = format_example_to_conversation_gemma(en_src, ru_trans, train=True)
            prompt = format_example_to_conversation_gemma(en_src, ru_trans, train=False)

        text = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        texts.append(text)
        prompts.append(prompt)

        if response_part:
            response_start_idx = text.find(response_part) + len(response_part)

            instruction_len = len(tokenizer.encode(text[:response_start_idx], add_special_tokens=False))
            response_len = len(tokenizer.encode(text[response_start_idx:], add_special_tokens=False))

            assert instruction_len + response_len == len(tokenizer.encode(text, add_special_tokens=False))

            label_mask = [0] * instruction_len + [1] * response_len

            label_masks.append(label_mask)

    if response_part:
        return {
            "text": texts,
            "prompt": prompts,
            "label_mask": label_masks
        }
    else:
        return {
            "text": texts,
            "prompt": prompts
        }
