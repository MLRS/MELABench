# <img src="logo.jpg" alt="MELABench logo" width="100px"> A Maltese Evaluation Language Benchmark 🇲🇹

MELABench is an evaluation benchmark of model capabilities on Maltese.
We have a publicly available benchmark which is periodically updated: https://huggingface.co/spaces/MLRS/MELABench

To run evaluation on this benchmark, we provide code to do this in various ways:
- [Prompting](prompting): runs models by prompting them with pre-defined instructions.
- [Fine-Tuning](finetuning): trains models first before evaluating them.

We also release small fine-tuned models on each task:
- BERTu: https://huggingface.co/collections/MLRS/bertu-683ac54c1b6ab3ae715cb43d
- mT5-Small: https://huggingface.co/collections/MLRS/mt5-small-683eecd001179a722c98298b 

## Citation

This work was introduced in [MELABenchv1: Benchmarking Large Language Models against Smaller Fine-Tuned Models for Low-Resource Maltese NLP](https://arxiv.org/abs/2506.04385).
Cite as follows:

```bibtex
@inproceedings{micallef-borg-2025-melabenchv1,
    title = "{MELAB}enchv1: Benchmarking Large Language Models against Smaller Fine-Tuned Models for Low-Resource {M}altese {NLP}",
    author = "Micallef, Kurt  and
      Borg, Claudia",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1053/",
    pages = "20505--20527",
    ISBN = "979-8-89176-256-5",
}
```
