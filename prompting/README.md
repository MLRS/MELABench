# Prompting

Prompt templates are defined in the [task configuration directory](tasks).

Experiments for zero/few-shot prompting can be run through [our fork of the LM Evaluation Harness](https://github.com/KurtMica/lm-evaluation-harness/) as follows:

```shell
lm-eval --model_args pretrained=${model} \
    --batch_size auto:4 --max_batch_size 128 \
    --tasks="${tasks}" --cache_requests="true" \
    --device cuda \
    --trust_remote_code \
    --write_out --log_samples --output_path=${OUTPUT_PATH} --seed=${seed} --include_path tasks/${version}
```

- `${task}` can be the task's identifier or an appropriate tag/group.
  When specifying `melabench` this runs all tasks in 0-shot & `melabench_fewshot` runs all tasks in 1-shot (excluding any tasks without training/development data to sample few-shots from).
  You can additionally override the number of shots by specifying the `--num_fewshot` argument.
- `${version}` determines the prompt templates to use.
  Currently, these are the initial prompts from v1.0: `1.0_english` (English instructions) or `1.0_maltese` (Maltese instruction).
