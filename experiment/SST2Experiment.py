from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from .AbstractExperiment import AbstractExperiment
from tqdm import tqdm
import numpy as np

class SST2Experiment(AbstractExperiment):
    """
    Experiment class for Children's Book Test task
    """
    def __init__(self, model, dataset, tokenizer, evaluate_set = "validation"):
        assert evaluate_set in ["train", "validation", "test"]
        
        super().__init__(model, dataset)
        self.tokenizer = tokenizer
        self.evaluate_set = evaluate_set
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def preprocess_data(self):
        """
        Process the dataset. Return a DataLoader for the experiment.

        :param dataset: The dataset to be processed.
        :return: A DataLoader object.
        """
        self.tokenizer.pad_token = self.tokenizer.eos_token
        def collate_fn(batch):
            """
            Given a single input in SST-2, return the list of 2 different options: "Positive" / "Negative"
            """
            sentence = batch[0]['sentence']
            options = ["positive", "negative"]
            answer = batch[0]['label']
            returned_list = []
            prompt = " The sentiment of this sentence is "
            for choice in options:
                curr_sentence = '"' + sentence + '"' + prompt + choice
                tokenized_sentence = self.tokenizer(curr_sentence, return_tensors="pt")
                returned_list.append(tokenized_sentence)

            return (returned_list, [1, 0], answer)
        
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
        self.model.eval()
        model_output = []
        answers = []

        for batch in tqdm(dataloader):
            scores = []
            tokenized_sentences, options, answer = batch
            for tokenized_sentence in tokenized_sentences:
                _input = tokenized_sentence.to(self.device)
                outputs = self.model(**_input, labels=_input['input_ids'])
                log_likelihood = -outputs.loss.item()
                prob = torch.exp(torch.tensor(log_likelihood)).item()
                scores.append(prob)
            model_output.append(options[np.argmax(scores)])
            answers.append(answer)
            
        return (model_output, answers)

    def postprocess(self, results):
        """
        calculate accuracy
        """
        model_output, answer = results
        correct_count = 0
        for i in range(len(model_output)):
            if model_output[i] == answer[i]:
                correct_count += 1
        return correct_count / len(model_output)
    

