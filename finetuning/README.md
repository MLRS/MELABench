# Fine-Tuning

The [`transformers` library](https://github.com/huggingface/transformers/) is used for training & evaluation of generative models.
The provided scripts allow training of generative models on any of the supported tasks.
_To fine-tune encoder-only models refer to the [BERTu repository](https://github.com/MLRS/BERTu/tree/2025.acl/finetune)._

For reference, the following `${dataset_args}` can be used for each task:
- Sentiment Analysis: `--train_file="https://raw.githubusercontent.com/jerbarnes/typology_of_crosslingual/master/data/sentiment/mt/train.csv" --validation_file="https://raw.githubusercontent.com/jerbarnes/typology_of_crosslingual/master/data/sentiment/mt/dev.csv" --test_file="https://raw.githubusercontent.com/jerbarnes/typology_of_crosslingual/master/data/sentiment/mt/test.csv" --source_column=text --target_column=label --target_column_mapping="{\"0\": \"negative\", \"1\": \"positive\"}"`
- SIB-200: `--dataset_name=Davlan/sib200 --dataset_config_name=mlt_Latn --source_column=text --target_column=category`
- Taxi1500: `--train_file="${data_base_path}/Taxi1500/mlt_Latn_train.tsv" --validation_file="${data_base_path}/Taxi1500/mlt_Latn_dev.tsv" --test_file="${data_base_path}/Taxi1500/mlt_Latn_test.tsv" --source_column=text --target_column=category`
- Maltese News Categories: `--dataset_name=MLRS/maltese_news_categories --source_column=text --target_column=labels`
- MultiEURLEX: `--dataset_name=coastalcph/multi_eurlex --dataset_config_name="mt" --trust_remote_code --source_column=text --target_column=labels --target_column_mapping="eurovoc_concepts"`
- OPUS-100: `--dataset_name=MLRS/OPUS-MT-EN-Fixed --source_column=en --target_column=mt`
- WebNLG: `--dataset_name=datasets/webnlg.py --dataset_config_name=mt --trust_remote_code=true --train_file="${data_base_path}/WebNLG/mt_train.xml" --validation_file="${data_base_path}/WebNLG/mt_dev.xml" --test_file="${data_base_path}/WebNLG/mt_test.xml" --source_column=modified_triple_sets --target_column=lex`
- EUR-Lex-Sum: `--dataset_name=dennlinger/eur-lex-sum --dataset_config_name=maltese --trust_remote_code=true --source_column=reference --target_column=summary`
- Maltese News Headlines: `--dataset_name=MLRS/maltese_news_headlines --source_column=text --target_column=title`


## Generative (NLG) Tasks

For generative tasks use the following script:

```shell
python run_seq2seq.py \
    -${dataset_args} \
    --model_name_or_path=${model} \
    --do_train --num_train_epochs=200 --early_stopping_patience=20 --per_device_train_batch_size=32 --max_source_length=256 --max_target_length=256 --learning_rate=1e-3 --optim=adafactor \
    --do_eval --predict_with_generate --eval_strategy=epoch --per_device_eval_batch_size=32 --metric_names="bleu,chrf,rouge" --load_best_model_at_end --metric_for_best_model=${metric} --greater_is_better=true \
    --do_predict --save_strategy=epoch --save_total_limit=20 --output_dir="${output_path}"
```

The `${metric}` used is task-dependent:
- OPUS-100, WebNLG: `chrf_score`
- EUR-Lex-Sum, Maltese News Headlines: `rouge_rougeL`


## Discriminative (NLU) Tasks

For discriminative tasks, training/evaluation is based on log-likelihoods of each label (not the actual generated output).
A different script is used, but most parameters are otherwise the same:

```shell
python run_seq2seq_classification.py \
    ${dataset_args} \
    --model_name_or_path=${model} \
    --do_train --num_train_epochs=200 --early_stopping_patience=20 --per_device_train_batch_size=32 --max_source_length=256 --max_target_length=256 --learning_rate=1e-3 --optim=adafactor \
    --do_eval --predict_with_generate --eval_strategy=epoch --per_device_eval_batch_size=32 --metric_names="f1" --metric_kwargs="{\"average\": \"macro\"}" --load_best_model_at_end --metric_for_best_model=f1_f1 --greater_is_better=true \
    --do_predict --save_strategy=epoch --save_total_limit=20 --output_dir="${output_path}"
```
