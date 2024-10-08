import argparse
import difflib
import json
import logging
import os
import re
import statistics
from pathlib import Path

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

    def estimate_prob(self, input_str, candidate):
        if (input_str, candidate) in self.memo:
            return self.memo[(input_str, candidate)]
        input_ids = self.tokenizer(input_str, return_tensors="pt",
                                   add_special_tokens=True)["input_ids"]
        #
        final_input_str = input_str + " " + f"[{candidate}]"
        with_candidate_ids = self.tokenizer(final_input_str, return_tensors="pt",
                                            add_special_tokens=True)["input_ids"]
        candidate_len = with_candidate_ids.shape[1] - input_ids.shape[1]
        labels = with_candidate_ids.clone()
        labels[:, :-candidate_len] = -100
        #
        with torch.no_grad():
            outputs = self.model(with_candidate_ids.cuda(), labels=labels)
            neg_log_likelihood = outputs.loss.item()
            self.memo[(input_str, candidate)] = neg_log_likelihood
        return neg_log_likelihood
    

    def evaluate(self, examples):
        preds = []
        for ex in tqdm(examples):
            raw_input = ex["input"]
            input_str = raw_input.rstrip()

            expected_output = ex["expected_output"]
            negative_output = ex["negative_output"]

            expected_output_loss = self.estimate_prob(input_str, expected_output)
            negative_output_loss = self.estimate_prob(input_str, negative_output)
            correct = expected_output_loss < negative_output_loss
            pred = {
                "example_id": ex["example_id"],
                "model_input": input_str,
                "correct": correct,
                "expected_output_loss": expected_output_loss,
                "negative_output_loss": negative_output_loss,
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
    dataset_id = "coref-data/pcr_qa_prompt"
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

    out_dir = Path(os.environ["SCRATCH"]) / "pcr_preds" / "qa"

    for config in config_names:
        fname = f"{config}_qa_local.jsonl"
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
