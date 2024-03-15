# SNIP: Machine Unlearning via Selective Neuron-wise Interpretable Pruning

This repository holds the code and data of SNIP: Machine Unlearning via Selective Neuron-wise Interpretable Pruning
## Method

![](method.png)

## Installation
1. Install Python (3.10)
2. Install remaining requirements using `pip install -r requirements.txt`

## Quick Start

### Fetch neuron explanations of GPT-2
run `python fetch_neuron_explanations.py`

### Create pruning dictionary

Follow the instructions in `Retrieve concepts and concepts embedding of D.ipynb` retrieve concept embeddings for GPT-2 MLP neurons and an arbitrary dataset D.

Follow the instructions in `Retrieve pruning neuron dict.ipynb` to get pruning neuron dictionary based on neuron importance scores.

### Experiments

run `python run.py --experiment_name [CBT_V/CBT_P] --model_name [gpt2] --rule [prune/None] --selected_neuron_path [Path to the pruning neuron dictionary]`

## Acknowledgements

- [Language models can explain neurons in language models](https://openai.com/research/language-models-can-explain-neurons-in-language-models)

