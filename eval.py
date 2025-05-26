import os
import argparse
from functools import partial
from statistics import mean

import evaluate
import comet

def read_file(file_path: str) -> list[str]:
    if not file_path:
        return []

    with open(file_path) as f:
        data = f.readlines()

    data = list(map(lambda line: line.strip(), data))

    return data

def check_valid_for_metrics(
    metric: str,
    sources: list[str],
    references: list[str],
    candidates: list[str]
) -> None:
    msg = "{} missing. Please specify with {} or make sure it's not empty."

    if metric in ['bleu', 'bleurt', 'chrf', 'chrf++']:
        assert references != [], msg.format("References", "'--ref_file'")
        assert candidates != [], msg.format("Candidates", "'--can_file'")
    elif metric == 'comet22':
        assert sources != [], msg.format("Sources", "'--src_file'")
        assert references != [], msg.format("References", "'--ref_file'")
        assert candidates != [], msg.format("Candidates", "'--can_file'")
    elif metric == 'cometkiwi22':
        assert sources != [], msg.format("Sources", "'--src_file'")
        assert candidates != [], msg.format("Candidates", "'--can_file'")

def format_for_metrics(
    metric: str,
    sources: list[str],
    references: list[str],
    candidates: list[str]
) -> tuple[list, list] | list[dict]:
    if metric in ['bleu', 'chrf', 'chrf++']:
        # maps to `list` of `list` of `str`
        refs = list(map(lambda ref_line: [ref_line], references))

        return refs, candidates

    elif metric == 'bleurt':
        return references, candidates

    elif metric == 'comet22':
        data = []

        for src, ref, can in zip(sources, references, candidates):
            data.append(
                {
                    'src': src,
                    'mt': can,
                    'ref': ref
                }
            )

        return (data, )

    elif metric == 'cometkiwi22':
        data = []

        for src, can in zip(sources, candidates):
            data.append(
                {
                    'src': src,
                    'mt': can
                }
            )

        return (data, )


def bleu(references: list[str], candidates: list[list[str]], debug: bool=False) -> float | tuple[float, dict]:
    _sacrebleu = evaluate.load('sacrebleu', cache_dir=os.environ['HF_HOME'])

    results = _sacrebleu.compute(predictions=candidates, references=references)

    if debug:
        return results['score'], results
    
    return results['score']

def bleurt(references: list[str], candidates: list[list[str]], debug: bool=False) -> float | tuple[float, dict]:
    _bleurt = evaluate.load('bleurt', config_name='BLEURT-20', cache_dir=os.environ['HF_HOME'])

    results = _bleurt.compute(predictions=candidates, references=references)

    if debug:
        return mean(results['scores']) * 100, results

    return mean(results['scores']) * 100

def chrf(references: list[str], candidates: list[list[str]], word_order: int, debug: bool=False) -> float | tuple[float, dict]:
    _chrf = evaluate.load('chrf', cache_dir=os.environ['HF_HOME'])

    results = _chrf.compute(predictions=candidates, references=references, word_order=word_order)

    if debug:
        return results['score'], results
    
    return results['score']

def comet22(data: list[dict], debug: bool=False):
    model_path = comet.download_model('Unbabel/wmt22-comet-da', saving_directory=os.environ['HF_HOME'])

    _comet = comet.load_from_checkpoint(model_path)

    results = _comet.predict(data, batch_size=16)

    if debug:
        return results.system_score * 100, results
    
    return results.system_score * 100

def cometkiwi22(data: list[dict], debug: bool=False):
    model_path = comet.download_model('Unbabel/wmt22-cometkiwi-da', saving_directory=os.environ['HF_HOME'])

    _cometkiwi = comet.load_from_checkpoint(model_path)

    results = _cometkiwi.predict(data, batch_size=16)

    if debug:
        return results.system_score * 100, results
    
    return results.system_score * 100


ALL_METRICS = ['bleu', 'chrf', 'chrf++', 'comet22', 'cometkiwi22']#, 'bleurt']
METRIC_MAP = {
    'bleu': bleu,
    # 'bleurt': bleurt,
    'chrf': partial(chrf, word_order=0),
    'chrf++': partial(chrf, word_order=2),
    'comet22': comet22,
    'cometkiwi22': cometkiwi22,
}

if __name__ == '__main__':
    # TODO: add support for 'metricx-23' and 'metricx-24'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--metrics',
        default=['all'],
        nargs='+',
        choices=ALL_METRICS + ['all'],
        help='Evaluation metrics.'
    )
    parser.add_argument('--src_file', type=str, default=None, help='Source file path.')
    parser.add_argument('--ref_file', type=str, default=None, help='Reference file path.')
    parser.add_argument('--can_file', type=str, default=None, help='Candidate file path.')
    parser.add_argument('--res_file', type=str, default='eval_results.txt', help='Evaluation result file path.')
    parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained evaluation models checkpoint directory.')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug.')

    args = parser.parse_args()

    args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
    os.environ['HF_HOME'] = args.checkpoints_dir

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from torch.cuda import is_available as cuda_is_available
    device = 'cuda' if cuda_is_available() else 'cpu'

    if args.debug:
        print(args)

    sources = read_file(args.src_file)
    references = read_file(args.ref_file)
    candidates = read_file(args.can_file)

    if 'all' in args.metrics:
        metrics = ALL_METRICS
    else:
        metrics = args.metrics

    res_f = open(args.res_file, 'w')

    for metric in metrics:
        print(f'Evaluating {metric.upper()} score')
        try:
            check_valid_for_metrics(metric, sources, references, candidates)

            data = format_for_metrics(metric, sources, references, candidates)

            # score = locals()[metric](*data)
            score = METRIC_MAP[metric](*data)

            print(f'{metric.upper()} score = {score}', file=res_f)

            print('Done\n')
        except AssertionError as e:
            print(e)
            print('Failed\n')
            continue

    res_f.close()

    print(f'Done. Evaluation results are saved at {os.path.abspath(args.res_file)}')