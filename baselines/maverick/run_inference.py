
from functools import partial
import datasets
import huggingface_hub
from maverick import Maverick
from tqdm import tqdm

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

def get_token_mapping(sentences, context_start, context_end):
    local_to_global = {}
    global_to_local = {}
    t = 0
    for s_i in range(context_start, context_end):
        sentence = sentences[s_i]
        for i in range(len(sentence["tokens"])):
            local_to_global[(s_i, i)] = t
            global_to_local[t] = (s_i, i)
            t += 1
    return local_to_global, global_to_local


def local_mention_to_global(local_to_global, mention):
    sent, start, end = mention
    return (
                local_to_global[(sent, start)],
                local_to_global[(sent, end)]
            )


def global_mention_to_local(global_to_local, mention):
    start, end = mention
    start_sent, start_tok = global_to_local[start]
    end_sent, end_tok = global_to_local[end]
    assert start_sent == end_sent and end_tok >= start_tok
    return [start_sent, start_tok, end_tok]


def run_model_inference(model, dataset, use_gold_mentions, include_speaker, use_local_context, singletons=False,
                        identifier=""):
    # skip if repo exists
    if huggingface_hub.repo_exists(repo_id="coref-data/maverick_preds", repo_type="dataset"):
        repo_info = huggingface_hub.repo_info(repo_id="coref-data/maverick_preds", repo_type="dataset")
        
        config_names = repo_info.card_data.config_names
        if not config_names:
            config_names = [x["config_name"] for x in repo_info.card_data.configs]

        if identifier in config_names:
            print("Inference already exists: ", identifier)
            return

    correct = 0
    total = 0
    outputs = []

    print(identifier)
    for ex in tqdm(dataset):
        sentences = ex["sentences"]

        context_start = 0
        context_end = len(sentences)

        if use_local_context:
            context_start = ex["local_context_start"]
            context_end = ex["local_context_end"]

        local_to_global, global_to_local = get_token_mapping(sentences, context_start, context_end)
        words = [[x["text"] for x in s["tokens"]] for s in sentences[context_start:context_end]]

        speakers = None
        if include_speaker:
            speakers = [[s["speaker"] if s["speaker"] is not None else ""]*len(s["tokens"])
                        for s in sentences[context_start:context_end]]
            
        mentions = None
        if use_gold_mentions:
            lm_to_global = partial(local_mention_to_global, local_to_global)
            mentions = [lm_to_global(ex["pronoun"]),
                        lm_to_global(ex["antecedents"][0]),
                        lm_to_global(ex["distractors"][0])]

        # hotfix
        words = [[w if w != "''" else '""' for w in s] for s in words]
        try:
            prediction = model.predict(words, speakers=speakers, predefined_mentions=mentions, singletons=singletons)
            
            predicted_clusters = prediction["clusters_token_offsets"]
            gm_to_local = partial(global_mention_to_local, global_to_local)
            predicted_clusters = [[gm_to_local(m) for m in c] for c in predicted_clusters]
        except:
            print("Failed to create valid prediction.")
            print(words, speakers, "\n", mentions, singletons)
            predicted_clusters = []

        output_example = {
            "id": ex["id"],
            "words": words,
            "speakers": speakers,
            "predefined_mentions": mentions,
            "singletons": singletons,
            "predicted_clusters": predicted_clusters,
        }
        outputs.append(output_example)

        total += 1
        if len(predicted_clusters) == 1 and len(predicted_clusters[0]) == 2:
                if ex["pronoun"] in predicted_clusters[0] and ex["antecedents"][0] in predicted_clusters[0]:
                    correct += 1
        elif len(predicted_clusters) == 2:
            predicted_clusters = sorted(predicted_clusters, key=lambda x: len(x))
            if len(predicted_clusters[0]) == 1 and len(predicted_clusters[1]) == 2:
                if ex["pronoun"] in predicted_clusters[1] and ex["antecedents"][0] in predicted_clusters[1]:
                    correct += 1
    print(correct, total, float(correct) / total)

    output_dataset = datasets.Dataset.from_list(outputs)
    output_dataset.push_to_hub("coref-data/maverick_preds", identifier, private=True)


def evaluate(model, dataset, singletons=False, identifier=""):
    for use_gold_mentions in [True, False]:
        for include_speaker in [True, False]:
            for use_local_context in [True, False]:
                new_identifier = "_".join([
                    identifier,
                    f"goldments={use_gold_mentions}",
                    f"speaker={include_speaker}",
                    f"localcontext={use_local_context}",
                    ])
                run_model_inference(model, dataset, use_gold_mentions, include_speaker,
                                    use_local_context, singletons, new_identifier)

def main():
    for training_set in ["ontonotes", "preco"]:
        model_name = f"sapienzanlp/maverick-mes-{training_set}"
        model = Maverick(hf_name_or_path=model_name, device="cuda")
        for config_name in config_names:
            dataset_name = "coref-data/pcr_single_antecedent"
            dataset = datasets.load_dataset(dataset_name, config_name)
            for split in ["validation", "test"]:
                if split not in dataset:
                    continue
                singletons = training_set == "preco"
                identifier = f"{config_name}_split={split}_train={training_set}"
                evaluate(model, dataset[split], singletons=singletons, identifier=identifier)

main()
