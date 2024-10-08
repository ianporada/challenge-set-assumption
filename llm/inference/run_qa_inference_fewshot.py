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


PROMPT = [{'role': 'user', 'content': 'Please carefully read the following passages. For each passage, you must identify which noun the mention marked in *bold* refers to.\n\nPassage: [Randy] was more of a baseball fan than [Michael] so *he* was heavily invested in every postseason .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to randy'}, {'role': 'user', 'content': 'Passage: [Christopher] helped [Randy] to obtain a federal employer identification number because *he* has more experience in that area .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to christopher'}, {'role': 'user', 'content': 'Passage: [Lawrence] refused to speak in front of the class even after being encouraged by [Nelson] , because *he* was n\'t convincing enough .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to nelson'}, {'role': 'user', 'content': 'Passage: [The invaders] showed no mercy to [the defenders] because *they* are a vicious and unforgiving people .\nQuestion: In the above passage, what does "*they*" refer to?'}, {'role': 'assistant', 'content': '*they* refers to the invaders'}, {'role': 'user', 'content': 'Passage: [Ian] began to test [Ryan] on various elements in the periodic table , because *he* had a test coming up .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to ryan'}, {'role': 'user', 'content': 'Passage: Learning to play a glockenspiel was instinctual for [Derrick] but not [Adam] because *he* is arty .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to derrick'}, {'role': 'user', 'content': 'Passage: In tense situations [Jeffrey] did not scare easily but [Jason] did because *he* was very cowardly .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to jason'}, {'role': 'user', 'content': 'Passage: [Dennis] explained the comic book cover to [Aaron] as *he* was unfamiliar with the whole series .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to aaron'}, {'role': 'user', 'content': 'Passage: [Christopher] had a grumbling stomach but not [Nelson] because *he* had had waited a shortened time for lunch .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to nelson'}, {'role': 'user', 'content': 'Passage: [Ian] reorganized the balls in [Derrick] \'s set triangle to put the 8 - ball in the center , since *he* misunderstood the rules .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to derrick'}, {'role': 'user', 'content': 'Passage: [The couple] took the car to [the shop] on Tuesday , but *they* will not have to pay until next week .\nQuestion: In the above passage, what does "*they*" refer to?'}, {'role': 'assistant', 'content': '*they* refers to the couple'}, {'role': 'user', 'content': 'Passage: [Christopher] got to the store faster than [Donald] although *he* rode on a horse instead of walking .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to donald'}, {'role': 'user', 'content': 'Passage: [Brett] was able to solve the problem that [Aaron] could not , so *he* was given a warning .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to aaron'}, {'role': 'user', 'content': 'Passage: [Robert] aimed his water gun at [Ian] and started to soak him in red paint because *he* wanted payback for a prank .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to robert'}, {'role': 'user', 'content': 'Passage: [The Apes] were determined to be contaminated by [disease specialists] , so *they* were forced to be released in secured captivity .\nQuestion: In the above passage, what does "*they*" refer to?'}, {'role': 'assistant', 'content': '*they* refers to the apes'}, {'role': 'user', 'content': 'Passage: [Logan] worked for the government while [Joseph] worked at a coffee shop , so *he* was paid by taxpayer money .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to logan'}, {'role': 'user', 'content': 'Passage: [Craig] went shopping at an expensive food store while [Kevin] went to the food kitchen since *he* was poor .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to kevin'}, {'role': 'user', 'content': 'Passage: I think that [Derrick] really hated yoga a lot more than [William] because *he* never went .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to derrick'}, {'role': 'user', 'content': 'Passage: After the terrible traffic accident , [Justin] paid 3 million to [Benjamin] , because *he* was severely injured .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to benjamin'}, {'role': 'user', 'content': 'Passage: [Groupon] lied about their earnings to [the stakeholders] , but *they* found out about the deception .\nQuestion: In the above passage, what does "*they*" refer to?'}, {'role': 'assistant', 'content': '*they* refers to the stakeholders'}, {'role': 'user', 'content': 'Passage: Drinking alcohol always causes [Brett] to lose control but not [Robert] , so *he* has to imbibe in moderation .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to brett'}, {'role': 'user', 'content': 'Passage: The products that [Steven] bought were more expensive than those of [Leslie] , since *he* was a wealthy person .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to steven'}, {'role': 'user', 'content': 'Passage: [Craig] did not share [Adam] \'s dream of competing in the Olympics , because *he* hated playing sports .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to craig'}, {'role': 'user', 'content': 'Passage: [Justin] helped [Kyle] in the struggle of finally getting his private pilot \'s license and *he* was happy for him .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to justin'}, {'role': 'user', 'content': 'Passage: [Kenneth] uses shampoo for color treated hair and [Lawrence] does not because *he* has dyed hair .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to kenneth'}, {'role': 'user', 'content': 'Passage: [Derrick] had more patience in the waiting room than [Eric] as , *he* is restless and impatient .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to eric'}, {'role': 'user', 'content': 'Passage: Creating a website was easier for [Christopher] than [Aaron] because *he* did n\'t know how to use computers .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to aaron'}, {'role': 'user', 'content': 'Passage: [Brian] was looking forward to having kids , but [Hunter] was not , because *he* had always wanted to be a parent .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to brian'}, {'role': 'user', 'content': 'Passage: [Benjamin] was asking [Hunter] for some tips caring for some sea monkeys , because *he* had no idea what to do .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to benjamin'}, {'role': 'user', 'content': 'Passage: [Brett] knew how to take care of a kitten better than [Neil] because *he* was a cat person .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to brett'}, {'role': 'user', 'content': 'Passage: [Jason] bought some fish at the fishmonger \'s shop to cook for [Hunter] , since *he* was his boyfriend .\nQuestion: In the above passage, what does "*he*" refer to?'}, {'role': 'assistant', 'content': '*he* refers to hunter'}, {'role': 'user', 'content': 'Passage: [The lifeguards] evacuated [the swimmers] from the public pool since *they* needed to clean the pool .\nQuestion: In the above passage, what does "*they*" refer to?'}, {'role': 'assistant', 'content': '*they* refers to the lifeguards'}]



def make_fewshot(raw_input):
    raw_input = raw_input[raw_input.find("Passage: "):]

    answer_start = raw_input.rfind("Answer: ")
    before_answer = raw_input[:answer_start]
    after_answer = raw_input[answer_start + len("Answer: "):]

    messages = [
        {"role": "user", "content": before_answer},
        {"role": "assistant", "content": after_answer.rstrip()},
    ]

    return PROMPT + messages


class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    def estimate_prob(self, input_str, candidate):
        input_ids = self.tokenizer(input_str, return_tensors="pt",
                                   add_special_tokens=False)["input_ids"]
        #
        final_input_str = input_str + " " + f"[{candidate}]" # + "." + self.tokenizer.eos_token # TODO no period?
        with_candidate_ids = self.tokenizer(final_input_str, return_tensors="pt",
                                            add_special_tokens=False)["input_ids"]
        candidate_len = with_candidate_ids.shape[1] - input_ids.shape[1]
        labels = with_candidate_ids.clone()
        labels[:, :-candidate_len] = -100
        #
        with torch.no_grad():
            outputs = self.model(with_candidate_ids.cuda(), labels=labels)
            neg_log_likelihood = outputs.loss.item()
        return neg_log_likelihood
    

    def evaluate(self, examples):
        preds = []
        for ex in tqdm(examples):
            raw_input = ex["input"]
            messages = make_fewshot(raw_input)
            input_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
            if input_str[-len(self.tokenizer.eos_token):] == self.tokenizer.eos_token:
                input_str = input_str[:-len(self.tokenizer.eos_token)].rstrip()
            
            expected_output_loss = self.estimate_prob(input_str, ex["expected_output"])
            negative_output_loss = self.estimate_prob(input_str, ex["negative_output"])
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

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    if args.llama2:
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    
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
            not ex["local_context"] and \
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

    out_dir = Path(os.environ["SCRATCH"]) / "pcr_preds" / "fs"

    for config in config_names:
        fname = f"{config}_fs_local.jsonl"
        if args.llama2:
            fname = "llama2_" + fname
        fpath = out_dir / fname
        if fpath.exists():
            continue
        
        logging.info(f"Running inference on: {config}")
        test_data = [ex for ex in data if ex["dataset"] == config]
        test_data = [ex for ex in test_data if valid_example(ex)]
        chunk_size = 500
        if len(test_data) > chunk_size:
            chunks = [test_data[i:i + chunk_size] for i in range(0, len(test_data), chunk_size)]
            for i, chunk in enumerate(chunks):
                fname = f"{config}_fs_local_{i+1}_of_{len(chunks)}.jsonl"
                fpath = out_dir / fname
                if fpath.exists():
                    continue
                preds = evaluator.evaluate(chunk)
                acc = statistics.mean([x["correct"] for x in preds])
                print(f"Accuracy on {config} ({i+1} of {len(chunks)}): {acc}")
                with open(fpath, "w") as of:
                    for x in preds:
                        of.write(json.dumps(x) + "\n")
            continue
        preds = evaluator.evaluate(test_data)
        acc = statistics.mean([x["correct"] for x in preds])
        print(f"Accuracy on {config}: {acc}")
        with open(fpath, "w") as of:
            for x in preds:
                of.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
