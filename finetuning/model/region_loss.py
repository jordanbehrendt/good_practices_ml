import torch
import ast
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score

class Regional_Loss(torch.nn.Module):
    def __init__(self, country_list):
        """
        Initializes the Regional_Loss object.

        Args:
            country_list (pandas.DataFrame): A DataFrame containing the infromation of country_list_region_and_continent.csv.

        Attributes:
            device (torch.device): The device (CPU or GPU) on which the model will be trained.
            country_list (pandas.DataFrame): The input country list DataFrame.
            country_dict (dict): A dictionary mapping country names to indices in the country_list.
            region_indices (list): A list of lists containing the indices of countries in each region.
            regions (torch.Tensor): A tensor containing the region indices for each country.
            regions_dict (dict): A dictionary mapping country indices to region indices.
            selective_sum_operator (torch.Tensor): A tensor used for selective sum operation.

        """

        super(Regional_Loss, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.country_list = country_list
        self.country_dict = {country: index for index,
                             country in enumerate(self.country_list["Country"])}
        self.region_indices = country_list.groupby('Intermediate Region Name')[
            'Country'].apply(lambda x: list(x.index)).to_list()
        self.regions = self.country_list["One Hot Region"].apply(lambda x: torch.argmax(
            torch.tensor(ast.literal_eval(x), dtype=torch.float32, device=self.device)))
        self.regions_dict = {index: region for index,
                             region in enumerate(self.regions)}
        self.selective_sum_operator = torch.zeros(len(self.region_indices), len(
            self.country_list), dtype=torch.float32, device=self.device)
        for i, indices in enumerate(self.region_indices):
            self.selective_sum_operator[i, indices] = 1
        self.selective_sum_operator

    def forward(self, outputs, targets):
        """
        Forward pass of the model.

        Args:
            outputs (torch.Tensor): The output tensor from the model.
            targets (list): The list of target values.

        Returns:
            tuple: A tuple containing the mean region loss and mean country loss.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the indices of all targets, corrseponding to the region index of the one hot encoded region vector
        target_region_enc = torch.tensor(
            [self.regions_dict[target] for target in target_countries_idxs], device=self.device)
        # sum the outputs of the countries in each region
        region_outputs = torch.matmul(
            outputs, self.selective_sum_operator.transpose(0, 1))

        country_loss = F.cross_entropy(outputs, torch.tensor(
            target_countries_idxs, device=self.device))
        region_loss = F.cross_entropy(region_outputs, target_region_enc)

        return region_loss.mean(), country_loss.mean()

    def claculate_region_accuracy(self, outputs, targets):
        """
        Calculates the accuracy of region predictions.

        Args:
            outputs (torch.Tensor): The output tensor from the model.
            targets (list): The list of target countries.

        Returns:
            torch.Tensor: The mean accuracy of region predictions.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the indices of all targets, corrseponding to the region index of the one hot encoded region vector
        target_region_idx = torch.tensor(
            [self.regions_dict[target] for target in target_countries_idxs], device=self.device)
        # sum the outputs of the countries in each region
        region_outputs = torch.matmul(
            outputs, self.selective_sum_operator.transpose(0, 1))

        region_predictions_idxs = torch.argmax(region_outputs, axis=1)
        # calculate the accuracy of the region predictions
        return torch.mean((region_predictions_idxs == target_region_idx).float())

    def calculate_country_accuracy(self, outputs, targets):
        """
        Calculates the accuracy of country predictions.

        Args:
            outputs (torch.Tensor): The predicted outputs of the model.
            targets (list): The target country labels.

        Returns:
            torch.Tensor: The mean accuracy of country predictions.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the index of the preidcted country
        country_predictions_idxs = torch.argmax(outputs, axis=1)
        # calculate the accuracy of the country predictions
        return torch.mean((country_predictions_idxs == torch.tensor(target_countries_idxs, device=self.device)).float())
    
    def calculate_metrics_per_class(self, outputs, targets):
        """
        Calculates precision, recall, F1-score, and support for country predictions for each class.

        Args:
            outputs (torch.Tensor): The predicted outputs of the model.
            targets (list): The target country labels.

        Returns:
            tuple: A tuple containing precision, recall, F1-score, and support for each class.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the index of the predicted country
        country_predictions_idxs = torch.argmax(outputs, axis=1).tolist()
        # calculate the precision, recall, F1-score, and support of the country predictions
        precision, recall, fscore, support = score(target_countries_idxs, country_predictions_idxs, zero_division=0)
        
        country_metrics_index = np.take(self.country_list["Country"].unique(), np.unique(target_countries_idxs))

        return precision, recall, fscore, support, country_metrics_index

    def calculate_metrics_per_region(self, outputs, targets):
        """
        Calculates precision, recall, F1-score, and support for region predictions for each class.

        Args:
            outputs (torch.Tensor): The predicted outputs of the model.
            targets (list): The target country labels.

        Returns:_
            tuple: A tuple containing precision, recall, F1-score, and support for each class.
        """
        # get the indices of all targets for the country_list, which is the index of the one hot encoded country vector
        target_countries_idxs = [self.country_dict[target]
                                 for target in targets]
        # get the indices of all targets, corrseponding to the region index of the one hot encoded region vector
        target_region_idx = torch.tensor(
            [self.regions_dict[target] for target in target_countries_idxs], device=self.device)
        # sum the outputs of the countries in each region
        region_outputs = torch.matmul(
            outputs, self.selective_sum_operator.transpose(0, 1))
        region_predictions_idxs = torch.argmax(region_outputs, axis=1).tolist()
        # calculate the precision, recall, F1-score, and support of the region predictions
        precision, recall, fscore, support = score(target_region_idx.tolist(), region_predictions_idxs, zero_division=0)
        all_regions = np.sort(self.country_list["Intermediate Region Name"].unique())

        region_metrics_index = np.unique(target_region_idx.tolist())
        region_metrics_index = np.take(all_regions, region_metrics_index)
        return precision, recall, fscore, support, region_metrics_index


    def calculate_mixed_metrics(self, outputs, targets):
        country_predictions_idxs = torch.argmax(outputs, axis=1)
        target_countries = [self.country_dict[target] for target in targets]

        region_outputs = torch.matmul(outputs, self.selective_sum_operator.transpose(0, 1))
        region_predictions_idxs = torch.argmax(region_outputs, axis=1)
        target_region_idx = torch.tensor([self.regions_dict[target.item()] for target in country_predictions_idxs], device=self.device)

        unique_countries = np.unique(np.concatenate([country_predictions_idxs.numpy(),target_countries]))
        TP, FP, FN = np.zeros(len(unique_countries)), np.zeros(len(unique_countries)), np.zeros(len(unique_countries))

        for idx, country in enumerate(unique_countries):
            true_pos = target_countries == country
            pred_pos = (country_predictions_idxs == country).numpy()
            region = self.regions_dict[country]
            true_region = (target_region_idx == region).numpy()
            pred_region = (region_predictions_idxs == region).numpy()

            TP[idx] = (true_pos & pred_pos).sum().item() + (((true_pos & ~pred_pos) & (pred_region & true_region)).sum().item() / 2)
            FP[idx] = (~true_pos & pred_pos).sum().item()
            FN[idx] = (true_pos & ~pred_pos & ~pred_region).sum().item()

        mixed_prec = TP / (TP + FP)
        mixed_rec = TP / (TP + FN)
        mixed_f1 = 2 * (mixed_prec * mixed_rec) / (mixed_prec + mixed_rec)

        mixed_prec = np.nan_to_num(mixed_prec, nan=0.0)
        mixed_rec = np.nan_to_num(mixed_rec, nan=0.0)
        mixed_f1 = np.nan_to_num(mixed_f1, nan=0.0)

        return mixed_prec, mixed_rec, mixed_f1
    

