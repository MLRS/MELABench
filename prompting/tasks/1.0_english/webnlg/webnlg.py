# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The WebNLG 2023 Challenge."""


import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import datasets


_HOMEPAGE = "https://synalp.gitlabpages.inria.fr/webnlg-challenge/challenge_2023/"

_DESCRIPTION = """\
The WebNLG challenge consists in mapping data to text. The training data consists
of Data/Text pairs where the data is a set of triples extracted from DBpedia and the text is a verbalisation
of these triples. For instance, given the 3 DBpedia triples shown in (a), the aim is to generate a text such as (b).

a. (John_E_Blaha birthDate 1942_08_26) (John_E_Blaha birthPlace San_Antonio) (John_E_Blaha occupation Fighter_pilot)
b. John E Blaha, born in San Antonio on 1942-08-26, worked as a fighter pilot

As the example illustrates, the task involves specific NLG subtasks such as sentence segmentation
(how to chunk the input data into sentences), lexicalisation (of the DBpedia properties),
aggregation (how to avoid repetitions) and surface realisation
(how to build a syntactically correct and natural sounding text).
"""

_LICENSE = ""

_CITATION = """\
@inproceedings{web_nlg,
  author    = {Claire Gardent and
               Anastasia Shimorina and
               Shashi Narayan and
               Laura Perez{-}Beltrachini},
  editor    = {Regina Barzilay and
               Min{-}Yen Kan},
  title     = {Creating Training Corpora for {NLG} Micro-Planners},
  booktitle = {Proceedings of the 55th Annual Meeting of the
               Association for Computational Linguistics,
               {ACL} 2017, Vancouver, Canada, July 30 - August 4,
               Volume 1: Long Papers},
  pages     = {179--188},
  publisher = {Association for Computational Linguistics},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-1017},
  doi       = {10.18653/v1/P17-1017}
}
"""

# From: https://github.com/WebNLG/2023-Challenge
_URL = "data.zip"

_LANGUAGES = ["br", "cy", "ga", "mt", "ru"]


def et_to_dict(tree):
    dct = {tree.tag: {} if tree.attrib else None}
    children = list(tree)
    if children:
        dd = defaultdict(list)
        for dc in map(et_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        dct = {tree.tag: dd}
    if tree.attrib:
        dct[tree.tag].update((k, v) for k, v in tree.attrib.items())
    if tree.text:
        text = tree.text.strip()
        if children or tree.attrib:
            if text:
                dct[tree.tag]["text"] = text
        else:
            dct[tree.tag] = text
    return dct


def parse_entry(entry, language=None):
    res = {}
    otriple_set_list = entry["originaltripleset"]
    res["original_triple_sets"] = [{"otriple_set": otriple_set["otriple"]} for otriple_set in otriple_set_list]
    mtriple_set_list = entry["modifiedtripleset"]
    res["modified_triple_sets"] = [{"mtriple_set": mtriple_set["mtriple"]} for mtriple_set in mtriple_set_list]
    res["category"] = entry["category"]
    res["eid"] = entry["eid"]
    res["size"] = int(entry["size"])
    res["lex"] = [
        {
            "comment": ex.get("comment", ""),
            "lid": ex.get("lid", ""),
            "text": ex.get("text", ""),
            "lang": ex.get("lang", ""),
        } for ex in entry.get("lex", []) if language is None or ex.get("lang", "") == language
    ]
    res["shape"] = entry.get("shape", "")
    res["shape_type"] = entry.get("shape_type", "")
    return res


def xml_file_to_examples(filename, language=None):
    tree = ET.parse(filename).getroot()
    examples = et_to_dict(tree)["benchmark"]["entries"][0]["entry"]
    return [parse_entry(entry, language) for entry in examples]


class Challenge2023(datasets.GeneratorBasedBuilder):
    """The WebNLG 2023 Challenge dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [datasets.BuilderConfig(name=language) for language in _LANGUAGES]

    def _info(self):
        features = datasets.Features(
            {
                "category": datasets.Value("string"),
                "size": datasets.Value("int32"),
                "eid": datasets.Value("string"),
                "original_triple_sets": datasets.Sequence(
                    {"otriple_set": datasets.Sequence(datasets.Value("string"))}
                ),
                "modified_triple_sets": datasets.Sequence(
                    {"mtriple_set": datasets.Sequence(datasets.Value("string"))}
                ),
                "shape": datasets.Value("string"),
                "shape_type": datasets.Value("string"),
                "lex": datasets.Sequence(
                    {
                        "comment": datasets.Value("string"),
                        "lid": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "lang": datasets.Value("string"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_files = self.config.data_files
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "xml_file": data_files["train"][0],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "xml_file": data_files["validation"][0],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "xml_file": data_files["test"][0],
                },
            ),
        ]

    def _generate_examples(self, xml_file):
        """Yields examples."""
        id_ = 0
        for exple_dict in xml_file_to_examples(xml_file, None if self.config.name == "default" else self.config.name):
            yield id_, exple_dict
            id_ += 1
