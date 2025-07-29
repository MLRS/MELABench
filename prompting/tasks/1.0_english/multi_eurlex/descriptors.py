import json
import urllib.request
from datasets import load_dataset


CLASS_LABEL = load_dataset("coastalcph/multi_eurlex", "all_languages", split="test").features["labels"].feature
with urllib.request.urlopen("https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json") as url:
    DESCRIPTORS =  json.loads(url.read())


def id_to_descriptor(id, language="mt"):
    global DESCRIPTORS
    return DESCRIPTORS[id][language]

def choices(doc):
    global CLASS_LABEL
    return [id_to_descriptor(label) for label in CLASS_LABEL.names]

def choice(doc):
    return [label for label in doc["labels"]]
