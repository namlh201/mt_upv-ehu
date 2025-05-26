import os
import argparse

from dotenv import load_dotenv
import comet

from eval import format_for_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--src_file', type=str, default=None, help='Source file path.')
parser.add_argument('--can_file', type=str, default=None, help='Candidate file path.')
parser.add_argument('--res_file', type=str, default='eval_results.txt', help='Evaluation result file path.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained evaluation models checkpoint directory.')

args = parser.parse_args()

load_dotenv()

def read_file(file_path: str) -> list[str]:
    if not file_path:
        return []

    with open(file_path) as f:
        data = f.readlines()

    data = list(map(lambda line: line.strip(), data))

    return data

def main(args: argparse.Namespace):
    en_srcs = read_file(os.path.join("data", args.src_file))
    ru_cans = read_file(os.path.join("data", args.can_file))

    model_path = comet.download_model('Unbabel/XCOMET-XL', saving_directory=os.environ['HF_HOME'])

    cometkiwi = comet.load_from_checkpoint(model_path)

    data_from_can = format_for_metrics('cometkiwi22', sources=en_srcs, references=[], candidates=ru_cans)[0]

    can_results = cometkiwi.predict(data_from_can, batch_size=40)

    can_scores = can_results.scores
    
    with open(os.path.join("data", args.res_file), "w") as f:
        for score in can_scores:
            print(score, file=f)

if __name__ == "__main__":
    main(args)
