from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

from .Metrics.eval import QGEvalCap
from argparse import ArgumentParser


def get_tuple(l):
    """
    return
    tails: list of string
    gens: list of string
    """
    gens = [str(l["generated"]).strip("'[]")]
    tails = [str(r).strip() for r in str(l["reference"]).split('|')]
    return {"tails": tails, "generations": gens}


def eval(model_name, data, k):
    topk_gts = {}
    topk_res = {}
    instances = []

    for i in range(len(data)):
        l = data.iloc[i]
        t = get_tuple(l)
        gens = t["generations"]
        tails = t["tails"]

        for (j, g) in enumerate(gens[:k]):

            instance = t.copy()
            instance["generation"] = g
            instances.append(instance)

            key = str(i) + "_" + str(j)
            topk_gts[key] = tails
            topk_res[key] = [g]

    QGEval = QGEvalCap(model_name, topk_gts, topk_res)
    score, scores = QGEval.evaluate()

    return score, scores, instances


def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--filename", default=None, type=str)
    args = parser.parse_args()

    input_file = f"results/{args.filename}"
    data = pd.read_csv(input_file)

    if 'reference' not in data.columns:
        reference_data = pd.read_csv('results/event_story_test_refs.csv')['context']
        reference_data = [' '.join(c.split('\t')[2:]) for c in reference_data]
        data['reference'] = reference_data
        data['generated'] = ['.'.join(g.split('.')[1:]) for g in data['generated'].values]
    model_name = '-'.join(args.filename.split('_')[:-2])

    if "v1" in args.filename:
        generated, reference = [], []
        for g, r in zip(data["generated"].values, data["reference"].values):
            if r != "end":
                generated.append(g)
                reference.append(r)
        data = pd.DataFrame(columns=["generated", "reference"])
        data["generated"] = generated
        data["reference"] = reference

    s, scores, instances = eval(model_name, data, 1)
    print(f"score:{s}")
    # print(f"instances 0-10: {instances[0:10]}")
