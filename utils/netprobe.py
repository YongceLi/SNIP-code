'''
Utilities for probing a torch model with a given dataset to get activation distribution for each neuron.
'''
import torch
import pickle
import numpy as np
from tqdm import tqdm

class NetProbe():
    """
    example usage:
    ```
        ...
        model = GPT2Model.from_pretrained('gpt2')
        ...
        netprobe_inst = NetProbe(model, "mlp.c_fc")
        netprobe_inst.add_hooks_to_store_act("all_stats") // or "mean_std"
        
        model inference ...
        
        act_stats = netprobe_inst.get_statistics()
        netprobe_inst.remove_all_hooks()
    ```
    """
    def __init__(self, model, key_module_names):
        self.model = model
        self.key_module_names = key_module_names
        self.statistics = {}
        self.handles = {}
        self.module_set = self.get_layer_names(model, key_module_names)

    def get_layer_names(self, model, key_module_names):
        """
        get the list of names of the layers with key_module_name as a substr in a given model
        """
        layer_name_list = []
        for name, module in model.named_modules():
            for key_module_name in key_module_names:
                if key_module_name in name:
                    layer_name_list.append(name)
                    break
        return layer_name_list

    def create_hook(self, layer_name, stats = "all_stats", type = "post"):
        """
        create hook function that specifically designed for given layer
        """
        def get_neuron_activations(module, input, output):
            if type == "pre":
                data = input[0]
            elif type == "post":
                data = output
            data_copy = data[0].cpu().numpy()
            if layer_name in self.statistics:
                self.statistics[layer_name].append(data_copy) # here can be modified to suit batched inputs
            else:
                self.statistics[layer_name] = [data_copy]
        def get_running_mean_std(module, input, output):
            if type == "pre":
                data = input[0]
            elif type == "post":
                data = output
            if layer_name not in self.statistics:
                self.statistics[layer_name] = {"sum": 0, "sum squares": 0, "n": 0, "mean": 0, "std": 0}
            data_copy = data[0].cpu().numpy()
            self.statistics[layer_name]["sum"] += np.sum(data_copy, axis = 0)
            self.statistics[layer_name]["sum squares"] += np.sum(data_copy ** 2, axis = 0)
            self.statistics[layer_name]["n"] += output[0].shape[0]
            self.statistics[layer_name]["mean"] = self.statistics[layer_name]["sum"] / self.statistics[layer_name]["n"]
            self.statistics[layer_name]["std"] = (self.statistics[layer_name]["sum squares"] / self.statistics[layer_name]["n"]
                                                    - self.statistics[layer_name]["sum"] * self.statistics[layer_name]["sum"]
                                                    / self.statistics[layer_name]["n"] / self.statistics[layer_name]["n"])
        if stats == "all_stats":
            return get_neuron_activations
        elif stats == "mean_std":
            return get_running_mean_std

    def add_hook(self, model, module_name, hook_func):
        """
        add the given hook function to a certain module of a given model
        """
        add_hook_statement = "model"
        module_name_split = module_name.split('.')
        for e in module_name_split:
            add_hook_statement += "."
            if e.isdigit():
                add_hook_statement = add_hook_statement[:-1] + f"[{e}]"
            else:
                add_hook_statement += e
        add_hook_statement += ".register_forward_hook(hook_func)"
        try:
            handle = eval(add_hook_statement)
        except Exception as e:
            print("add_hook Error: {e}")
            
        return handle
        
    def add_hooks_to_store_act(self, stats = "all_stats"):
        """
        add hooks to modules in the given module_set to store pre-activation neuron values
        """
        for e in self.module_set:
            if e in self.handles:
                raise ValueError(f"{e} already had a hook.")
            if "attn" in e or "down" in e:
                store_activation_hook = self.create_hook(e, stats = stats, type = "pre")
            else:
                store_activation_hook = self.create_hook(e, stats = stats, type = "post")
            self.handles[e] = self.add_hook(self.model, e, store_activation_hook)
        
    def remove_all_hooks(self):
        """
        remove all hooks
        """
        for key in self.handles:
            handle = self.handles[key]
            handle.remove()
        self.handles = {}

    def get_statistics(self):
        """
        get activation dictionary
        """
        return self.statistics
    
    def statistics_dump(self, file_path):
        """
        dump activation values
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.statistics, file)

