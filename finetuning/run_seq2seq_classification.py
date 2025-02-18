#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence (classification tasks).

Adapted from
- https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
- https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import json
import logging
import os
import sys
import urllib
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import ClassLabel, load_dataset
from filelock import FileLock
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

# Sometimes users will pass in a `str` repr of a dict in the CLI
# We need to track what fields those can be. Each time a new arg
# has a dict type, it must be added to this list.
# Important: These should be typed with Optional[Union[dict,str,...]]
_VALID_DICT_FIELDS = [
    "target_column_mapping",
    "metric_kwargs",
    "dataset_kwargs",
]


def _convert_str_dict(passed_value: dict):
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            # First check for bool and convert
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            # Check for digit
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                passed_value[key] = float(value)

    return passed_value


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_kwargs: Optional[Union[str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the dataset such as {'names': ['label', 'text']} for specifying column names."
            )
        },
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the source texts."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the target texts."},
    )
    target_column_mapping: Optional[Union[str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "A mapping specifying how labels should be represented in textual format. "
                "Any missing labels will be mapped as is."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_names: Optional[str] = field(
        default=None,
        metadata={"help": "The metrics to use for evaluation. Multiple metrics should be separated by commas."},
    )
    metric_kwargs: Optional[Union[str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the metric such as {'averge': 'macro'} for macro-averaging F1."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.target_column_mapping == "eurovoc_concepts":
            with urllib.request.urlopen("https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json") as url:
                self.target_column_mapping = {id: descriptors.get("mt", descriptors.get("en")) for id, descriptors in json.loads(url.read()).items()}

        # Parse in args that could be `dict` sent in from the CLI as a string
        for field in _VALID_DICT_FIELDS:
            passed_value = getattr(self, field)
            # We only want to do this if the str starts with a bracket to indiciate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, field, loaded_dict)

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class Seq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    early_stopping_patience: int = field(
        default=20,
        metadata={
            "help": "The number of epochs to perform when no improvement is made on the validation set,"
                    "before terminating the training procedure early."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "google-t5/t5-small",
        "google-t5/t5-base",
        "google-t5/t5-large",
        "google-t5/t5-3b",
        "google-t5/t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to Maltese: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the source texts and the second column for the
    # target texts (unless you specify column names for this with the `source_column` and `target_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_files=data_files or None,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            **data_args.dataset_kwargs,
        )
    else:
        dataset_kwargs = data_args.dataset_kwargs
        if extension == "tsv":
            builder_name = "csv"  # the "csv" builder reads both .csv and .tsv file
            dataset_kwargs = {"delimiter": "\t", **dataset_kwargs}
        elif extension == "jsonl":
            builder_name = "json"  # the "json" builder reads both .json and .jsonl files
        else:
            builder_name = extension  # e.g. "parquet"
        raw_datasets = load_dataset(
            builder_name,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_kwargs,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # We set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    is_multi_label = False
    if raw_datasets["train"].features[data_args.target_column].dtype == "list":  # multi-label classification
        is_multi_label = True
        logger.info("Label type is list, doing multi-label classification")

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    # Get the column names for input/target.
    if data_args.source_column is None:
        source_column = column_names[0]
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = column_names[1]
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    # In the event the labels are not a `ClassLabel`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            label = label if isinstance(label, list) else [label]
            unique_labels = unique_labels | set(label)
        label_list = [str(label) for label in unique_labels]
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    if (is_multi_label and isinstance(features[target_column].feature, ClassLabel)) or isinstance(
            features[target_column], ClassLabel):
        label_list = features[target_column].feature.names if is_multi_label else features[target_column].names
    else:
        label_list = get_label_list(raw_datasets["train"][target_column])
    label_to_text = {l: data_args.target_column_mapping.get(str(l), l) for l in label_list}
    label_to_id = {l: i for i, l in enumerate(label_to_text.values())}
    num_labels = len(label_list)

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        def convert_to_text(label):
            if isinstance(label, int) and label not in label_to_text:
                label = label_list[label]
            return label_to_text[str(label)]

        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] and examples[target_column][i] is not None:
                inputs.append(examples[source_column][i])
                target = examples[target_column][i]
                target = ", ".join([convert_to_text(label) for label in target]) if is_multi_label else convert_to_text(
                    target)
                targets.append(target)

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["__labels__"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            train_dataset = train_dataset.rename_column("__labels__",
                                                        "labels")  # workaround for cases when target_column is called "labels"

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.rename_column("__labels__",
                                                      "labels")  # workaround for cases when target_column is called "labels"

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            predict_dataset = predict_dataset.rename_column("__labels__",
                                                            "labels")  # workaround for cases when target_column is called "labels"

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    if data_args.metric_names:
        experiment_id = (model_args.model_name_or_path or "").replace("/", "__") + "+" + (
                    data_args.dataset_name or "").replace("/", "__")
        metric_modules = {}
        for metric_name in data_args.metric_names.split(","):
            metric_modules[metric_name] = evaluate.load(metric_name,
                                                        config_name="multilabel" if is_multi_label else None,
                                                        cache_dir=model_args.cache_dir, experiment_id=experiment_id)

    def postprocess_text(preds, labels):
        def convert_to_id(label):
            return label_to_id.get(label.strip(), -1)

        preds = [[convert_to_id(p) for p in pred.split(",")] if is_multi_label else convert_to_id(pred) for pred in preds]
        labels = [[convert_to_id(l) for l in label.split(",")] if is_multi_label else convert_to_id(label) for label in labels]

        return preds, labels
    
    def extract_loglikelihood_predictions(logits, labels):
        batch_size = logits.shape[0]
        num_candidates = len(label_to_id)
        tokenized_labels = tokenizer(list(label_to_id.keys()), padding=True, return_tensors="pt").input_ids

        # Convert logits to log probabilities
        log_probs = F.log_softmax(torch.tensor(logits), dim=-1)  # shape: (batch, seq_length, vocab_size)
        
        # Create a tensor to hold scores for each candidate per example
        # candidate_scores shape: (batch_size, num_candidates)
        candidate_scores = torch.zeros(batch_size, num_candidates)
        
        # Compute score for each candidate: the sum of log probabilities over the candidate tokens
        for i, candidate in enumerate(tokenized_labels):
            # Convert candidate token ids to tensor (shape: (1, cand_length))
            cand_tensor = torch.tensor(candidate).unsqueeze(0)
            cand_length = cand_tensor.size(1)
            
            # Expand candidate to all examples: (batch_size, cand_length)
            cand_expanded = cand_tensor.expand(batch_size, -1)
            
            # Here we assume that the logits corresponding to the label generation are in the first cand_length positions.
            # If your model produces logits for the label part separately (or if you have an offset), adjust accordingly.
            candidate_log_probs = torch.gather(
                log_probs[:, :cand_length, :],  # shape: (batch, cand_length, vocab_size)
                dim=-1,
                index=cand_expanded.unsqueeze(-1)
            ).squeeze(-1)  # shape: (batch, cand_length)
            
            # Create a mask in case the candidate contains any padding tokens
            mask = (cand_expanded != tokenizer.pad_token_id).float()
            candidate_log_probs = candidate_log_probs * mask
            
            # Sum the log probabilities for the candidate tokens to get a total score
            candidate_total_log_prob = candidate_log_probs.sum(dim=-1)  # shape: (batch,)
            candidate_scores[:, i] = candidate_total_log_prob
        
        # Decide on predicted labels:
        # If a threshold is provided, select all candidates with scores above or equal to the threshold.
        # Otherwise, use top-K selection where K is the number of true labels (if provided) per example.
        decoded_preds = []
        
        # Ensure that true_labels is a list (or array) of lists/sets per example.
        # Here we assume true_labels is provided in such a format.
        for i, scores in enumerate(candidate_scores.tolist()):
            if is_multi_label:
                K = len(labels[i])
                selected_indices = np.argsort(scores)[-K:]
                selected_indices = set(selected_indices)
            else:
                selected_indices = np.argmax(scores)
            decoded_preds.append(selected_indices)
        
        return decoded_preds

    def compute_metrics(eval_pred):
        """
        Computes metrics based on the log-likelihood of label tokens.
        
        Args:
            eval_pred (Tuple[torch.Tensor, torch.Tensor]): A tuple where:
                - logits: The raw logits output by the model of shape (batch_size, seq_length, vocab_size).
                - labels: The tokenized labels of shape (batch_size, seq_length). Padding tokens should be set to -100 or another ignore_index.
        
        Returns:
            dict: A dictionary with metrics, including average log-likelihood and exact match accuracy.
        """
        logits, labels = eval_pred  # unpack predictions and labels
        if isinstance(logits, tuple):
            logits = logits[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        _, decoded_labels = postprocess_text([], decoded_labels)

        decoded_preds = extract_loglikelihood_predictions(logits, decoded_labels)

        if is_multi_label:
            one_hot_encoder = MultiLabelBinarizer(classes=list(label_to_id.values()))
            decoded_preds = one_hot_encoder.fit_transform(decoded_preds)
            decoded_labels = one_hot_encoder.fit_transform(decoded_labels)

        results = {}
        for metric_name, metric in metric_modules.items():
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, **data_args.metric_kwargs)
            for key, value in result.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        results[f"{metric_name}_{key}_{n}"] = v
                elif not isinstance(value, list):
                    results[f"{metric_name}_{key}"] = value

        return results

    class Seq2SeqClassificationTrainer(Seq2SeqTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            labels = inputs.get("labels")
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else None
                logits = outputs.logits
            return (loss, logits, labels)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Initialize our Trainer
    trainer = Seq2SeqClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                decoded_labels = tokenizer.batch_decode(predict_dataset["labels"], skip_special_tokens=True)
                _, decoded_labels = postprocess_text([], decoded_labels)
                predictions = extract_loglikelihood_predictions(predict_results.predictions, decoded_labels)
                predictions = [",".join([label_list[p] for p in pred]) if is_multi_label else label_list[pred] for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "seq2seq"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
