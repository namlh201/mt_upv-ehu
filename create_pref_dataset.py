import os
import json

from datasets import Dataset
import numpy as np

def main_pref():
    candidate_models = ["aya-expanse-8b", "EuroLLM-9B-Instruct", "TowerInstruct-13B-v0.1", "nllb_3.3B", "ref"]

    with open("data/res_ref/src.en") as f:
        en_src_list = list(map(lambda line: line.strip(), f.readlines()))

    ru_trans_list_list = []
    xcomet_scores_list_list = []

    for candidate_model in candidate_models:
        with open(f"data/res_{candidate_model}/can.ru") as f:
            ru_trans_list = list(map(lambda line: line.strip(), f.readlines()))
            ru_trans_list_list.append(ru_trans_list)

        with open(f"data/res_{candidate_model}/scores.txt") as f:
            xcomet_scores_list = list(map(float, f.readlines()))
            xcomet_scores_list_list.append(xcomet_scores_list)
        
    xcomet_scores_list_list = np.asarray(xcomet_scores_list_list).T

    chosen_model_indices = np.argmax(xcomet_scores_list_list, axis=-1)
    rejected_model_indices = np.argmin(xcomet_scores_list_list, axis=-1)

    pref_dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    chosen_stats = { candidate_model: 0 for candidate_model in candidate_models }
    rejected_stats = { candidate_model: 0 for candidate_model in candidate_models }

    for i in range(len(en_src_list)):
        prompt = {
            "role": "user",
            "content": f"Translate the following text from English to Russian.\nEnglish: {en_src_list[i]}\nRussian: "
        }
        
        chosen = {
            "role": "assistant",
            "content": ru_trans_list_list[chosen_model_indices[i]][i]
        }
        chosen_stats[candidate_models[chosen_model_indices[i]]] += 1

        rejected = {
            "role": "assistant",
            "content": ru_trans_list_list[rejected_model_indices[i]][i]
        }
        rejected_stats[candidate_models[rejected_model_indices[i]]] += 1

        pref_dataset["prompt"].append(prompt)
        pref_dataset["chosen"].append(chosen)
        pref_dataset["rejected"].append(rejected)

    print("Chosen statistics:")
    print(json.dumps(chosen_stats))

    print("Rejected statistics:")
    print(json.dumps(rejected_stats))

    pref_dataset = Dataset.from_dict(pref_dataset)
    
    os.makedirs("data/news-commentary-v18_pref", exist_ok=True)

    pref_dataset.to_json("data/news-commentary-v18_pref/news-commentary-v18.en-ru.jsonl")

def main_sft():
    candidate_models = ["aya-expanse-8b", "EuroLLM-9B-Instruct", "TowerInstruct-13B-v0.1", "nllb_3.3B", "ref"]

    with open("data/res_ref/src.en") as f:
        en_src_list = list(map(lambda line: line.strip(), f.readlines()))

    ru_trans_list_list = []
    xcomet_scores_list_list = []

    for candidate_model in candidate_models:
        with open(f"data/res_{candidate_model}/can.ru") as f:
            ru_trans_list = list(map(lambda line: line.strip(), f.readlines()))
            ru_trans_list_list.append(ru_trans_list)

        with open(f"data/res_{candidate_model}/scores.txt") as f:
            xcomet_scores_list = list(map(float, f.readlines()))
            xcomet_scores_list_list.append(xcomet_scores_list)
        
    xcomet_scores_list_list = np.asarray(xcomet_scores_list_list).T

    chosen_model_indices = np.argmax(xcomet_scores_list_list, axis=-1)

    sft_dataset = {
        "en": [],
        "ru": []
    }

    for i in range(len(en_src_list)):
        sft_dataset["en"].append(en_src_list[i])
        sft_dataset["ru"].append(ru_trans_list_list[chosen_model_indices[i]][i])

    sft_dataset = Dataset.from_dict(sft_dataset)
    
    os.makedirs("data/news-commentary-v18_sft", exist_ok=True)

    sft_dataset.to_json("data/news-commentary-v18_sft/news-commentary-v18.en-ru.jsonl")

if __name__ == "__main__":
    main_pref()
    main_sft()
