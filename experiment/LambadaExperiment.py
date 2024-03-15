from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from .AbstractExperiment import AbstractExperiment
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

class LambadaExperiment(AbstractExperiment):
    """
    Experiment class for Lambada task
    """
    def __init__(self, model, dataset, tokenizer, evaluate_set = "test", batch_size = 20, ignore_fragments = True):
        assert evaluate_set in ["train", "validation", "test"]
        
        super().__init__(model, dataset)
        self.tokenizer = tokenizer
        self.evaluate_set = evaluate_set
        self.batch_size = batch_size
        self.ignore_fragments = ignore_fragments
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def preprocess_data(self):
        """
        Process the dataset. Return a DataLoader for the experiment.

        :param dataset: The dataset to be processed.
        :return: A DataLoader object.
        """
        def preprocess(text):
            # preprocess text 
            text = text.replace("“", '"')
            text = text.replace("”", '"')
            text = text.replace("''", '"')
            text = text.replace("``", '"')
            return text
        
        def collate_fn(batch):
            """
            Given a single input in SST-2, return the list of 2 different options: "Positive" / "Negative" 1/-1
            """
            encoded_last_word = []
            actual_last_word = []
            fragments = []
            batch_encoded = []
            for element in batch:
                element['text'] = preprocess(element['text'])
                element['text'] = '\n' + element['text'].strip()
                sentence = element['text']
                line_encoded = self.tokenizer.encode(sentence)
                encoded_last_word.append(self.tokenizer.decode(line_encoded[-1:]).strip())
                actual_last_word.append(sentence.split()[-1].strip())
                if encoded_last_word != actual_last_word:
                    fragments.append(True)
                else:
                    fragments.append(False)
                batch_encoded.append(line_encoded)

            max_len = max(len(encoded) for encoded in batch_encoded)
            batch_padded = []
            lengths = [] 
            for encoded in batch_encoded:
                batch_padded.append(encoded+[0]*(max_len - len(encoded)))
                lengths.append(len(encoded))
            
            batch_padded = torch.tensor(batch_padded)
            batch_padded = batch_padded.to(self.device)

            return (batch_padded, batch_encoded, encoded_last_word, actual_last_word, fragments, lengths)
        
        dataloader = DataLoader(self.dataset[self.evaluate_set], batch_size=1, collate_fn=collate_fn)
        return dataloader

    @torch.no_grad()
    def run(self, dataloader):
        """
        Evaluate the model on a certain task with the given DataLoader.

        :param model: The ML model to be evaluated.
        :param dataloader: The DataLoader to provide data.
        :return: Results of the model evaluation.
        """
        self.model.to(self.device)
        model_outputs = []
        actual_results = []

        for (batch_padded, batch_encoded, encoded_last_word, actual_last_word, fragments, lengths) in tqdm(dataloader):
            outputs = self.model(batch_padded)
            logits = outputs.logits
            for i in range(self.batch_size):
                if i >= len(batch_padded):
                    break
                last_idx = lengths[i] - 1
                observed = batch_encoded[i][last_idx]
                predicted = int(torch.argmax(logits[i][last_idx - 1]).item())
                if self.ignore_fragments and fragments[i]:
                    continue
                actual_results.append(observed)
                model_outputs.append(predicted)
            
        return (model_outputs, actual_results)

    def postprocess(self, results):
        """
        calculate metric
        """
        model_outputs, actual_results = results
        accuracy = accuracy_score(model_outputs, actual_results)
        return f"accuracy: {accuracy}"   
