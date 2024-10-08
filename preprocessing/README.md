# Preprocessing

## Initial Dataset Setup

Datasets are originally sourced from https://github.com/ianporada/coref-data

In addition, Winogrande was further preprocessed to generate a validation split and remove examples with non-person pronouns: [initial_dataset_setup/create_pronominal_winogrande.ipynb](initial_dataset_setup/create_pronominal_winogrande.ipynb).

## Creation of PCR instances

First, [convert_to_pronoun_instances.ipynb](convert_to_pronoun_instances.ipynb) converts existing dataset instances into the pronominal test instances used in the experiments

Next, [preprocessing/format_pronoun_instances.ipynb](preprocessing/format_pronoun_instances.ipynb) shuffles the distractor candidates and filters the datasets down to only those instances with a single antecedent within the context.