from tqdm import tqdm
import pickle
import numpy as np
import torch
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import math
import random

layer_pre_postfix = {
    'gemma': ['model.layers.', '.mlp.down_proj'],
    'gpt2-large': ['transformer.h.', '.mlp.c_fc'],
    'gpt2': ['h.', '.mlp.c_fc'],
    'distilbert': ['distilbert.transformer.layer.', '.ffn.lin1']
}
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_neuron_act_stats(layer_id, neuron_id, act_stats, model):
    global layer_pre_postfix
    stats_for_neuron = np.array([])
    values = act_stats[f'{layer_pre_postfix[model][0]}{layer_id}{layer_pre_postfix[model][1]}']
    for e in values:
        stats_for_neuron = np.concatenate((stats_for_neuron, e[:,neuron_id]), axis=0)
    return stats_for_neuron

def plot_distribution(stats1, stats2, title_layer_id, title_neuron_id, label1 = None, label2 = None, alpha = 0.5, color1 = 'blue', bins = 30, color2 = 'red', edgecolor = 'black', density = True, xlabel='Value', ylabel="Frequency"):
    plt.hist(stats1, label = label1, alpha=alpha, color=color1, bins=bins, edgecolor=edgecolor, density=density)
    plt.hist(stats2, label = label2, alpha=alpha, color=color2, bins=bins, edgecolor=edgecolor, density=density)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f"Distribution of Layer {title_layer_id}, Neuron {title_neuron_id}'s Pre-Activation Value")
    plt.show()

def get_KS_statistics(sample_list_1, sample_list_2):
    statistic, p_value = ks_2samp(sample_list_1, sample_list_2)
    return statistic, p_value

def transform_distribution(data, mu1, sigma1, mu2, sigma2):
    """
    Transform data from N(mu1, sigma1) to N(mu2, sigma2).

    Parameters:
    data (array-like): The original data sampled from N(mu1, sigma1).
    mu1 (float): The mean of the original distribution.
    sigma1 (float): The standard deviation of the original distribution.
    mu2 (float): The mean of the target distribution.
    sigma2 (float): The standard deviation of the target distribution.

    Returns:
    array-like: Transformed data sampled from N(mu2, sigma2).
    """
    standardized_data = (data - mu1) / sigma1
    transformed_data = standardized_data * sigma2 + mu2
    return transformed_data

def create_net_adjust_hookfunc(layer_info, func = "prune", type = "post"):
    '''
    adjust selected neurons from layers.
    '''
    def prune_neurons(module, input, output):
        data = output
        selected_neuron = layer_info["neuron_lst"]
        data[:, :, selected_neuron] = 0.0
        return data

    if func == "prune":
        return prune_neurons
    
def net_hook_neurons(model, selected_neuron_info_lst, func = "prune"):
    handles = []
    for module_name in selected_neuron_info_lst:
        add_hook_statement = "model"
        module_name_split = module_name.split('.')
        for e in module_name_split:
            add_hook_statement += "."
            if e.isdigit():
                add_hook_statement = add_hook_statement[:-1] + f"[{e}]"
            else:
                add_hook_statement += e
        if "attn" in add_hook_statement:
            layer_info = selected_neuron_info_lst[module_name]
            hook_func = create_net_adjust_hookfunc(layer_info, func = func, type = "pre")
            add_hook_statement += ".register_forward_pre_hook(hook_func)"
        elif "mlp" in add_hook_statement:
            layer_info = selected_neuron_info_lst[module_name]
            hook_func = create_net_adjust_hookfunc(layer_info, func = func, type = "post")
            add_hook_statement += ".register_forward_hook(hook_func)"
        try:
            handle = eval(add_hook_statement)
            handles.append(handle)
        except Exception as e:
            print("add_hook Error: {e}")
        
    return handles