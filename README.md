# The Challenge Set Assumption

Code for the paper "Solving the Challenge Set without Solving the Task: On Winograd Schemas as a Test of Pronominal Coreference Resolution" presented at CoNLL 2024.

## Environment

The precise package versions used are listed in the `requirements.txt`.

## Data Preprocessing

As a starting point, we use the formatted datasets made available from https://github.com/ianporada/coref-data

See [preprocessing/](preprocessing/) for the scripts used to preprocess these datasets.

The resulting formatted data is available as jsonlines upon request (due to licensing requirements). The permissive license subset of the data is publicly available at [coref-data/pcr_public_datasets](https://huggingface.co/datasets/coref-data/pcr_public_datasets/).

## Model Predictions

Model predictions (baselines and LMs) are available in jsonlines format at [coref-data/pcr_model_preds](https://huggingface.co/datasets/coref-data/pcr_model_preds).

## Model Weights

For almost all models, we use the publicly released weights. The finetuned Llama 3.1 8B model weights are available on Hugging Face here: https://huggingface.co/ianporada/llama-3.1-8b-pcr

## LM Predictions

LM inference and preprocessing code is located at [llm/](llm/)

## Baseline Predictions

Baseline inference code (dcoref and Maverick) is available at [baselines/](baselines/)

## Reference

```bibtex
TODO
```
