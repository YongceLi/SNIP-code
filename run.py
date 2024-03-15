import argparse
from utils import *
from experiment import *
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import torch
import pickle

def load_model_n_tokenizer(model_name):
    if "gpt" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name, return_dict=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer
    if "distilbert-sst2" in model_name:
        model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        return model, tokenizer
    return None, None

def main(args):

    experiment_name = args.experiment_name
    model_name = args.model_name
    rule = args.rule
    selected_neuron_path = args.selected_neuron_path

    model, tokenizer = load_model_n_tokenizer(model_name)
    
    if (rule != None) and (selected_neuron_path != None):
        with open(selected_neuron_path, 'rb') as f:
            selected_neuron_dict = pickle.load(f)

    if rule == "prune":
        model = InstrumentedModel(model) 
        model.prune_selected_neurons(selected_neuron_dict)
    if rule == "adjust":
        model = InstrumentedModel(model)
        model.adjust_selected_neurons(selected_neuron_dict)

    if experiment_name == "CBT_P":
        dataset = load_dataset("cbt", "P")
        task = CBTExperiment(model, dataset, tokenizer)
    elif experiment_name == "CBT_V":
        dataset = load_dataset("cbt", "V")
        task = CBTExperiment(model, dataset, tokenizer)
    elif experiment_name == "SST_2":
        dataset = load_dataset("sst2")
        task = SST2Experiment(model, dataset, tokenizer)
    elif experiment_name == "SST_2_GPT":
        dataset = load_dataset("sst2")
        task = SST2Experiment_GPT(model, dataset, tokenizer)
    elif experiment_name == "LAMBADA":
        dataset = load_dataset("lambada")
        task = LambadaExperiment(model, dataset, tokenizer)
    
    dataloader = task.preprocess_data()
    (model_output, answer) = task.run(dataloader)
    result = task.postprocess((model_output, answer))
    
    print(result)

    dump_file_path = f'{experiment_name}_{model_name}_{rule}.pkl'
    with open(dump_file_path, "wb") as fp:   
        pickle.dump((model_output, answer), fp)

    if rule is not None:
        model.close()
    

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser to run experiments")

    parser.add_argument('--experiment_name', choices=['CBT_P', 'CBT_V', 'SST_2', 'SST_2_GPT', 'LAMBADA'], default='CBT_P', help='The type of experiment')
    parser.add_argument('--model_name', choices=['gpt2', 'gpt2-medium', 'gpt2-l', 'gpt2-xl', 'distilbert-sst2'], default='gpt2', help='model name')
    parser.add_argument('--rule', choices=[None, 'prune', 'adjust'], default=None, help='Rule for manipulating neuron')
    parser.add_argument('--selected_neuron_path', type=str, default=None, help='The path to the dictionary of selected neurons to be manipulated')
    
    args = parser.parse_args()

    main(args)