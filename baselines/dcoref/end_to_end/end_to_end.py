import argparse
import os
import subprocess
from pathlib import Path

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


def run_inference(ifname, ofname):
    """
    For each file:
        1. copy to directory
        2. run inference
        3. copy outputs to new directory
    """
    subprocess.run("rm conlloutput-* ; rm evaluation_*", shell=True)

    temp_data_dir = Path(os.environ["SLURM_TMPDIR"]) / "data"
    subprocess.run(f"rm -rf {temp_data_dir}/*", shell=True)
    subprocess.run(f"cp {ifname} {temp_data_dir}", shell=True)

    subprocess.run("java -cp \"*\" edu.stanford.nlp.dcoref.SieveCoreferenceSystem -props \"inference.properties\"", shell=True)

    subprocess.run(f"cp conlloutput-* {ofname}", shell=True)
    subprocess.run(f"cp evaluation_* {ofname}", shell=True)


def write_dataset_as_conll(dataset_name, config_name):
    base_path = Path(os.environ["SCRATCH"]) / "pcr"
    dataset_path: Path = base_path / config_name

    for split in ["validation", "test"]:
        for use_local_context in [False, True]:
            ifname = dataset_path / f"split={split}_localcontext={use_local_context}.v4_auto_conll"
            if ifname.exists():
                ofname = dataset_path / "inference" / f"split={split}_localcontext={use_local_context}"
                if not ofname.exists():
                    os.makedirs(ofname)
                if len(os.listdir(ofname)) == 0:
                    run_inference(ifname, ofname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_index', type=int, choices=range(15))
    args = parser.parse_args()
    
    write_dataset_as_conll(dataset_name, config_names[args.dataset_index])


if __name__ == '__main__':
    main()
