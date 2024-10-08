import argparse
import difflib
import json
import logging
import os
import re
from pathlib import Path
import statistics

import datasets
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        self.memo = {}

    def clean(self, x):
        # x = x[:x.find(".")] if ("." in x) and ("." not in gt) else x
        stop_words = ["a", "an", "the", "that", "this"]
        x = " ".join([w for w in x.split() if w.lower() not in stop_words])
        x = x[:x.find("\n")] if "\n" in x else x
        return re.sub(re.compile(r'\s+'), '', x.lower().replace("[", "").replace("]", "").replace("*", ""))

    def evaluate(self, examples):
        preds = []
        for ex in tqdm(examples):
            raw_input = ex["input"]
            pronoun_str = None
            for w in ex["passage_words"][::-1]:
                if w[0] == w[-1] and w[0] == "*":
                    pronoun_str = w
                    break
            input_str = raw_input + f"The pronoun {pronoun_str} refers to"
            if input_str in self.memo:
                outputs = self.memo[input_str]
            else:
                # to speed things up, only generate up to possible output
                # (since we are looking at exact match of prefix anyway)
                lens = [
                    len(self.tokenizer(x, add_special_tokens=False)["input_ids"])
                    for x in [ex["expected_output"], ex["negative_output"]]
                ]
                max_new_tokens = max(lens) + 3
                outputs = self.pipeline(
                    input_str,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1, # greedy decoding
                    temperature=None,
                    return_full_text=False,
                    top_p=None,
                )
                self.memo[input_str] = outputs
            continuation = outputs[0]["generated_text"].lstrip()
            est = continuation
            gt = ex["expected_output"]
            dist = ex["negative_output"]
            correct = self.clean(gt) in self.clean(est) and \
                      self.clean(dist) not in self.clean(est)
            pred = {
                "example_id": ex["example_id"],
                "model_input": input_str, 
                "model_output": continuation,
                "correct": correct,
            }
            preds.append(pred)
        return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama2", help="Use older Llama 2 version.",
                    action="store_true")
    args = parser.parse_args()

    model_id = "meta-llama/Meta-Llama-3.1-8B"
    if args.llama2:
        model_id = "meta-llama/Llama-2-7b-hf"
    
    logging.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                device_map="auto", torch_dtype=torch.bfloat16)
    
    # make sure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # download data
    dataset_id = "coref-data/pcr_gpt3_prompt"
    logging.info(f"Loading dataset: {dataset_id}")
    dataset = datasets.load_dataset(dataset_id)
    data = dataset["train"].to_list()

    def valid_example(ex):
        return ex["split"] == "test" and \
            ex["local_context"] and \
            not ex["include_speaker"]
    
    config_names = [
        'conll2012_indiscrim_english_v4',
        'gum_indiscrim_ontogum',
        'arrau_indiscrim_default',
        'gap_indiscrim_default',
        'davis_pdp_indiscrim_default',
        'preco_indiscrim_default',
        'litbank_indiscrim_split_0',
        'gum_indiscrim_original',
        'phrase_detectives_indiscrim_default',
        'mmc_indiscrim_mmc_en',
        'davis_wsc_indiscrim_wsc273',
        'superglue_wsc_indiscrim_default',
        'dpr_indiscrim_default',
        'knowref_60k_indiscrim_default',
        'pronominal_winogrande_default'
    ]

    evaluator = Evaluator(model, tokenizer)

    out_dir = Path(os.environ["SCRATCH"]) / "pcr_preds" / "gpt3"

    for config in config_names:
        fname = f"{config}_gpt3_local.jsonl"
        if args.llama2:
            fname = "llama2_" + fname
        fpath = out_dir / fname
        if fpath.exists():
            continue
        
        logging.info(f"Running inference on: {config}")
        test_data = [ex for ex in data if ex["dataset"] == config]
        test_data = [ex for ex in test_data if valid_example(ex)]
        preds = evaluator.evaluate(test_data)
        acc = statistics.mean([x["correct"] for x in preds])
        print(f"Accuracy on {config}: {acc}")
        with open(fpath, "w") as of:
            for x in preds:
                of.write(json.dumps(x) + "\n")

if __name__ == "__main__":
    main()
