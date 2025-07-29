import numpy as np
import evaluate
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def macro_f1(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return f1_score(golds, preds, average="macro")


def multilabel_macro_f1(items):
    golds, preds = [], []
    classes = set()
    for labels, predictions in items:
        classes.update(labels)
        golds.append(labels)
        ranked_predictions = np.argsort(predictions)[::-1]
        top_predictions = ranked_predictions[:len(labels)]
        preds.append(top_predictions)

    binarizer = MultiLabelBinarizer(classes=list(classes))
    golds = binarizer.fit_transform(golds)
    preds = binarizer.fit_transform(preds)
    return f1_score(golds, preds, average="macro")


def mean_rprecision(items):
    p_ks = []
    for y_t, y_s in items:
        # if np.sum(y_t == 1):
        # p_ks.append(default_rprecision_score(y_t, y_s))
        p_ks.append(default_rprecision_score(y_t, y_s))

    return np.mean(p_ks)


def default_rprecision_score(y_true, y_score):
    unique_y = np.unique(y_true)

    # if len(unique_y) == 1:
    #     raise ValueError("The score cannot be approximated.")
    # elif len(unique_y) > 2:
    #     raise ValueError("Only supported for two relevant levels.")

    # pos_label = unique_y[1]
    # print("pos_label:", pos_label)
    # n_pos = np.sum(y_true == pos_label)
    # print("n_pos:", n_pos)
    pos_label = y_true
    n_pos = len(y_true)

    order = np.argsort(y_score)[::-1]
    # y_true = np.take(y_true, order[:n_pos])
    # print("take:", y_true)
    # n_relevant = np.sum(y_true == pos_label)
    # print("n_relevant:", n_relevant)
    n_relevant = 0
    for label in order[:n_pos]:
        if label in y_true:
            n_relevant += 1

    # Divide by n_pos such that the best achievable score is always 1.0.
    return float(n_relevant) / n_pos

def rouge(items):
    return items

def rougeL(items):
    unzipped_list = list(zip(*items))
    refs = unzipped_list[0]
    preds = unzipped_list[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rougeL"]
