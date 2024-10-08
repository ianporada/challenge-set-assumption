import os
from pathlib import Path

import datasets
import torch
import transformers
from transformers import TrainingArguments, pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

dataset  = datasets.load_dataset("coref-data/pcr_qa_prompt_training") # tokenize to generate input_ids and labels
model = pipeline.model
tokenizer = pipeline.tokenizer

tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(example):
    return example["text"]

response_template = " refer to?\nAnswer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

tmp_dir = Path(os.environ["SLURM_TMPDIR"])
training_args = TrainingArguments(
    output_dir = tmp_dir / "results",
    num_train_epochs = 10,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 32,
    learning_rate=2e-5,  
    warmup_steps=1000,
    lr_scheduler_type="inverse_sqrt",
    logging_dir = tmp_dir / "logs",
    logging_steps=10,
    eval_steps=100,
    save_steps=1000,
    eval_strategy="steps",
    do_eval=True,
    eval_on_start=True,
)

trainer = SFTTrainer(
    model=pipeline.model,
    tokenizer=pipeline.tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
