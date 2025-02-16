# Prompting

Prompt templates are defined in the [task configuration directory](tasks).

Experiments for zero/few-shot prompting can be run through [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/) as follows:

```shell
lm-eval --model_args pretrained=${model} \
    --batch_size auto:4 --max_batch_size 128 \
    --tasks="${tasks}" --cache_requests="true" \
    --num_fewshot=${shots} \
    --device cuda \
    --trust_remote_code \
    --write_out --log_samples --output_path=output/${shots}_shot --seed=${seed} --include_path tasks
```

The `${task}` can be the task's identifier or an appropriate tag.
The available tags are based on whether few-shot is possible (if the task has training data) & on the prompting language (English by default): `zero_shot`, `few_shot`, `zero_shot_maltese_instruction`, and `few_shot_maltese_instruction`.
