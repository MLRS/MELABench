from datasets import load_dataset

CLASS_LABEL = load_dataset("MLRS/maltese_news_categories", split="test").features["labels"].feature


def choices(doc):
    global CLASS_LABEL
    return CLASS_LABEL.names

def choice(doc):
    return [label for label in doc["labels"]]
