from transformers import PreTrainedTokenizer

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

def format_example_to_conversation_llama(en_src: str, ru_trans: str, train: bool=True) -> list[dict[str, str]]:
    # system_instruction = {
    #     "role": "system",
    #     "content": "You are a translation assistant specifically designed to provide accurate and contextually appropriate translations. Please ensure that your translation accurately captures the meaning and nuances of the English text, while also taking into account any cultural or linguistic differences that may apply."
    # }

    user_input = {
        "role": "user",
        "content": f"Translate the following text from English to Russian.\nEnglish: {en_src}\nRussian: "
    }

    assistant_response = {
        "role": "assistant",
        "content": ru_trans
    }

    if train:
        # conversation = [system_instruction, user_input, assistant_response]
        conversation = [user_input, assistant_response]
    else:
        # conversation = [system_instruction, user_input]
        conversation = [user_input]
    
    return conversation

def format_example_to_conversation_gemma(en_src: str, ru_trans: str, train: bool=True) -> list[dict[str, str]]:
    # system_instruction = {
    #     "role": "system",
    #     "content": "You are a translation assistant specifically designed to provide accurate and contextually appropriate translations. Please ensure that your translation accurately captures the meaning and nuances of the English text, while also taking into account any cultural or linguistic differences that may apply."
    # }

    user_input = {
        "role": "user",
        "content": f"Translate the following text from English to Russian. Do not give any explanation, only return the translated Russian text.\nEnglish: {en_src}\nRussian: "
    }

    assistant_response = {
        "role": "assistant",
        "content": ru_trans
    }

    if train:
        # conversation = [system_instruction, user_input, assistant_response]
        conversation = [user_input, assistant_response]
    else:
        # conversation = [system_instruction, user_input]
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

    # if not for_grpo:
    for en_src, ru_trans in zip(en_src_list, ru_trans_list):
        # en_src = ' '.join(en_src.split('\n\n'))
        # ru_trans = ' '.join(ru_trans.split('\n\n'))

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

            # instruction_len = len(tokenizer.encode(text[:response_start_idx]))
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