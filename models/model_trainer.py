import sys
sys.path.append('.')
sys.path.append('./scripts')
# ----------------------------------------------
import datetime
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
from models import nn
import os
from scripts import load_dataset, geo_metrics
from models.region_loss import Regional_Loss
import ast
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
import seaborn as sn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import yaml
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import time
import multiprocessing



class ModelTrainer():

    def __init__(self, model: torch.nn.Module, train_dataframe, country_list, region_list, num_folds=10, num_epochs=3, learning_rate=0.001, starting_regional_loss_portion=0.9, regional_loss_decline=0.2, train_dataset_name="Balanced", batch_size=260, seed=123) -> None:
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        torch.manual_seed(seed)
        self.training_dataset_name = train_dataset_name
        self.train_dataframe = train_dataframe
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.country_list = pd.read_csv(country_list)
        self.region_list = pd.read_csv(region_list, delimiter=',')
        self.regional_ordering_index = [8, 11, 144, 3, 4, 12, 16, 26, 28, 44, 46, 51, 52, 66, 74, 83, 95, 101, 105, 109, 121, 128, 153, 180, 191, 201, 202, 32, 43, 77, 81, 134, 140, 146, 179, 99, 106, 185, 187, 198, 58, 98, 122, 131, 133, 136, 159, 163, 166, 177, 178, 193, 195, 209, 210, 41, 80, 97, 102, 103, 126, 127, 192, 20, 31, 48, 84, 119, 152, 160, 162, 173, 194, 60, 137, 149, 165, 204, 78, 156, 7, 34, 35, 40, 64, 53, 56, 116, 117, 167, 188, 23, 33, 72, 196, 13, 50, 55, 59, 62, 65, 69,
                                        86, 88, 92, 94, 113, 115, 142, 168, 172, 38, 148, 189, 205, 9, 25, 27, 39, 42, 54, 61, 68, 76, 79, 147, 157, 197, 200, 24, 85, 100, 107, 125, 135, 150, 169, 184, 186, 203, 30, 138, 182, 208, 2, 17, 29, 89, 91, 111, 132, 143, 151, 0, 5, 15, 57, 71, 75, 82, 93, 120, 123, 130, 155, 161, 171, 175, 199, 206, 19, 22, 37, 45, 70, 73, 112, 124, 129, 139, 170, 174, 176, 183, 1, 6, 14, 21, 47, 67, 87, 90, 96, 104, 108, 145, 154, 158, 164, 181, 190, 207, 10, 18, 36, 49, 63, 110, 114, 118, 141]
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = Regional_Loss(self.country_list)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_count = 0
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.regional_portion = starting_regional_loss_portion
        self.regional_loss_decline = regional_loss_decline
        self.batch_size = batch_size

        # self.region_criterion = Regional_Loss(self.country_list, self.region_list)
        self.writer = SummaryWriter(
            log_dir=f'runs/seed_{seed}/{self.training_dataset_name[:-4]}/starting_regional_loss_portion-{starting_regional_loss_portion}/regional_loss_decline-{regional_loss_decline}/{self.timestamp}')
        self.start_training()

    def add_metrics_and_plot_tb(self, outputs, targets, name, step):
        per_class_precision, per_class_recall, per_class_f1, _, target_idx = self.criterion.calculate_metrics_per_class(outputs, targets)
        self.writer.add_scalar(f'{name} avg Class Precision', per_class_precision.mean(), step)
        self.writer.add_scalar(f'{name} avg Class Recall', per_class_recall.mean(), step)
        self.writer.add_scalar(f'{name} avg Class F1', per_class_f1.mean(), step)

        metrics_df = pd.DataFrame({'Precision': per_class_precision, 'Recall': per_class_recall, 'Fscore': per_class_f1})
        metrics_df.index = target_idx
        
        ignored_classes = metrics_df[(metrics_df['Precision'] == 0) & (metrics_df['Recall'] == 0)]
        metrics_df = metrics_df.drop(ignored_classes.index)

        bar_plot = metrics_df.plot(kind='bar', xlabel='Class', ylabel='Metrics', title='Metrics per Country, {len(ignored_classes)} ignored countries.').get_figure()
        self.writer.add_figure(f'{name} Metrics per Country', bar_plot, step)
        self.writer.add_scalar(f'{name} Number of Ignored Classes', len(ignored_classes), step)

        # Calculate metrics per region
        per_region_precision, per_region_recall, per_region_f1, _, region_index = self.criterion.calculate_metrics_per_region(outputs, targets)
        self.writer.add_scalar(f'{name} avg Region Precision', per_region_precision.mean(), step)
        self.writer.add_scalar(f'{name} avg Region Recall', per_region_recall.mean(), step)
        self.writer.add_scalar(f'{name} avg Region F1', per_region_f1.mean(), step)
        region_metrics_df = pd.DataFrame({'Precision': per_region_precision, 'Recall': per_region_recall, 'Fscore': per_region_f1})


        region_metrics_df.index = region_index
        
        ignored_regions = region_metrics_df[(region_metrics_df['Precision'] == 0) & (region_metrics_df['Recall'] == 0)]
        region_metrics_df = region_metrics_df.drop(ignored_regions.index)

        region_bar_plot = region_metrics_df.plot(kind='bar', xlabel='Region', ylabel='Metrics', title=f'Metrics per Region, {len(ignored_regions)} ignored regions.').get_figure()
        self.writer.add_figure(f'{name} Metrics per Region', region_bar_plot, step)
        self.writer.add_scalar(f'{name} Number of Ignored Regions', len(ignored_regions), step)
        self.writer.add_text(f'{name} List of Ignored Regions', ';'.join(ignored_regions.index.to_list()), step)
        self.writer.flush()


    def train_one_fold(self, train_loader):
        """Train one Epoch of the model. Based on Pytorch Tutorial.

        Args:
            epoch_index (int): Current epoch
            tb_writer (orch.utils.tensorboard.writer.SummaryWriter): Tensorboard wirter

        Returns:
            float: Average loss for the epoch
        """
        running_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_loader):

            # Zero gradients for every batch!
            self.optimizer.zero_grad()
            # Every data instance is an input + label pair
            inputs, labels = data
            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            regional_loss, country_loss = self.criterion(outputs, labels)
            loss = self.calculate_weighted_loss(regional_loss, country_loss)

            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            self.batch_count += 1

            # print(f"batch {i} loss: {loss}")
            # if i % 10 == 9:
            #     last_loss = running_loss/
            # print(f"batch {i} loss: {loss}")
            # if i % 10 == 9:
            #     last_lo
        fold_mean_loss = running_loss / len(train_loader)
        return fold_mean_loss

    def validate(self, epoch_index, fold_index, validation_dataset):
        inputs, targets = validation_dataset[:]
        outputs = self.model(inputs)
        
        avg_validation_region_accuracy = self.criterion.claculate_region_accuracy(
            outputs, targets)
        avg_validation_accuracy = self.criterion.calculate_country_accuracy(
            outputs, targets)
        

        # avg_validation_loss = validation_loss / len(validation_loader)
        print('Epoch {} Fold {} Validation Accuracy: {}, Validation Regional Accuracy: {}'.format(
            epoch_index + 1, fold_index + 1, avg_validation_accuracy, avg_validation_region_accuracy))
        self.writer.add_scalar(
            'Validation Accuracy', avg_validation_accuracy, epoch_index*self.num_folds + fold_index)
        self.writer.add_scalar('Validation Regional Accuracy',
                               avg_validation_region_accuracy, epoch_index*self.num_folds + fold_index)
        try:
            self.add_metrics_and_plot_tb(outputs, targets, "Epoch Validation", epoch_index*self.num_folds + fold_index)
        except Exception as e:
            print(e)
        
        return targets, outputs
        # self.writer.add_scalar('Validation Loss', avg_validation_loss, epoch_index*self.num_folds + fold_index)

        # torch.save(self.model.state_dict(),f'saved_models/model_{self.training_dataset_name}_epoch_{epoch_index}_batch_{i}')

    def start_training(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        validation_size = math.floor(
            len(self.train_dataframe.index) / self.num_folds)
        for epoch_index in range(self.num_epochs):
            targets = []
            outputs = torch.tensor([], dtype=torch.float32, device=self.device)
            if epoch_index > 0:
                self.regional_portion = self.regional_loss_decline * self.regional_portion
            for fold_index in range(self.num_folds):
                self.model.train()  # Set the model to training mode
                fold_validation_df = pd.DataFrame()
                drop_indices = []
                if fold_index == self.num_folds - 1:
                    fold_validation_df = self.train_dataframe.iloc[fold_index*validation_size:]
                    drop_indices = range(
                        fold_index*validation_size, len(self.train_dataframe.index))
                else:
                    fold_validation_df = self.train_dataframe.iloc[fold_index*validation_size: (
                        fold_index+1)*validation_size]
                    drop_indices = range(
                        fold_index*validation_size, (fold_index+1)*validation_size)

                fold_training_df = self.train_dataframe.drop(drop_indices)
                train_dataset = load_dataset.EmbeddingDataset_from_df(
                    fold_training_df, 'train')
                train_loader = DataLoader(
                    train_dataset,batch_size=self.batch_size, shuffle=False)
                avg_training_loss = self.train_one_fold(train_loader)
                self.writer.add_scalar(
                    'Training Loss', avg_training_loss, epoch_index*self.num_folds + fold_index)

                self.model.eval()  # Set the model to evaluation mode

                validation_dataset = load_dataset.EmbeddingDataset_from_df(
                    fold_validation_df, 'validation')
                # validation_loader = DataLoader(validation_dataset, shuffle=False)
                target, prediction = self.validate(epoch_index, fold_index, validation_dataset)
                targets.extend(target)
                outputs = torch.cat((outputs, prediction), dim=0)

                
                # self.writer.add_scalars('Training vs. Validation Loss',
                #         { 'Training' : avg_training_loss, 'Validation' : avg_validation_loss },
                #         epoch_index*self.num_folds + fold_index + 1)
                # print(f"Epoch [{epoch_index+1}/{self.num_epochs}] - Fold [{fold_index+1}/{self.num_folds}] - Average Train Loss: {avg_training_loss:.4f} - Val Loss: {avg_validation_loss:.4f}")
                self.writer.flush()
            if fold_index == self.num_folds - 1:
                with torch.no_grad():
                    predicitions_idx = torch.argmax(outputs, axis=1).tolist()
                    target_idx = [self.country_list[self.country_list['Country'] == target].index[0] for target in targets]
                    try:
                        self.add_metrics_and_plot_tb(outputs, targets, "Epoch Validation", epoch_index*self.num_folds)
                    except Exception as e:
                        print(e)
                    self.createConfusionMatrix(target_idx, predicitions_idx, "Validation Confusion Matrix", epoch_index*self.num_folds)
        
            torch.save(self.model.state_dict(),
                       f'saved_models/model_{self.training_dataset_name}_{timestamp}_epoch_{epoch_index+1}')


    def test_model(self, test_dataset):
        inputs, targets = test_dataset[:]
        outputs = self.model(inputs)
        
        avg_test_region_accuracy = self.criterion.claculate_region_accuracy(
            outputs, targets)
        avg_test_accuracy = self.criterion.calculate_country_accuracy(
            outputs, targets)
        
        self.writer.add_scalar(
            'Test Accuracy', avg_test_accuracy, )
        self.writer.add_scalar('Test Regional Accuracy',
                               avg_test_region_accuracy)

        with torch.no_grad():
            predicitions_idx = torch.argmax(outputs, axis=1).tolist()
            target_idx = [self.country_list[self.country_list['Country'] == target].index[0] for target in targets]
            self.createConfusionMatrix(target_idx, predicitions_idx, "Test Confusion Matrix", None)
            try:
                self.add_metrics_and_plot_tb(outputs, targets, "Test", None)
            except Exception as e:
                print(e)
        print('Training Dataset {} Test Accuracy: {}, Test Regional Accuracy: {}'.format(
            self.training_dataset_name, avg_test_accuracy, avg_test_region_accuracy))



    def createConfusionMatrix(self, true_countries, predicted_countries, figure_label, index):
        """
        Creates and visualizes the confusion matrix for country and region predictions.

        Args:
            true_countries (list): List of true country labels.
            predicted_countries (list): List of predicted country labels.
            figure_label (str): Label for the generated figures.
            index (int): Index for adding figures to the writer.

        Returns:
            None
        """
        # constant for classes
        classes = self.country_list['Country']
        np_classes = np.array(classes)

        # Build country confusion matrix
        cf_matrix = confusion_matrix(
            true_countries, predicted_countries, labels=range(0, 211))
        ordered_index = np.argsort(-cf_matrix.diagonal())
        ordered_matrix = cf_matrix[ordered_index][:, ordered_index]

        regionally_ordered_matrix = cf_matrix[self.regional_ordering_index][:,
                                                                            self.regional_ordering_index]

        ordered_classes = np_classes[ordered_index]
        regionally_ordered_classes = np_classes[self.regional_ordering_index]

        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
        ordered_df_cm = pd.DataFrame(
            ordered_matrix, index=ordered_classes, columns=ordered_classes)
        regionally_ordered_df_cm = pd.DataFrame(
            regionally_ordered_matrix, index=regionally_ordered_classes, columns=regionally_ordered_classes)

        np_regions = np.sort(
            np.array(list(set(self.country_list['Intermediate Region Name']))))

        # Build region confusion matrix
        true_regions = []
        predicted_regions = []
        for i in range(0, len(true_countries)):
            true_regions.append(ast.literal_eval(
                self.country_list.iloc[true_countries[i]]["One Hot Region"]).index(1))
            predicted_regions.append(ast.literal_eval(
                self.country_list.iloc[predicted_countries[i]]["One Hot Region"]).index(1))

        regions_cf_matrix = confusion_matrix(
            true_regions, predicted_regions, labels=range(0, len(np_regions)))
        regions_ordered_index = np.argsort(-regions_cf_matrix.diagonal())
        regions_ordered_matrix = regions_cf_matrix[regions_ordered_index][:,
                                                                          regions_ordered_index]

        ordered_regions = np_regions[regions_ordered_index]

        regions_df_cm = pd.DataFrame(
            regions_cf_matrix, index=np_regions, columns=np_regions)
        regions_ordered_df_cm = pd.DataFrame(
            regions_ordered_matrix, index=ordered_regions, columns=ordered_regions)
        plt.figure(1, figsize=(120, 70))
        figure = sn.heatmap(df_cm, cmap=sn.cubehelix_palette(
            as_cmap=True)).get_figure()
        plt.figure(2, figsize=(120, 70))
        ordered_figure = sn.heatmap(
            ordered_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        plt.figure(3, figsize=(120, 70))
        regionally_ordered_figure = sn.heatmap(
            regionally_ordered_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        plt.figure(4, figsize=(120, 70))
        regions_figure = sn.heatmap(
            regions_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        plt.figure(5, figsize=(120, 70))
        regions_ordered_figure = sn.heatmap(
            regions_ordered_df_cm, cmap=sn.cubehelix_palette(as_cmap=True)).get_figure()
        if (index == None):
            self.writer.add_figure(f"{figure_label}-unordered", figure)
            self.writer.add_figure(f"{figure_label}-ordered", ordered_figure)
            self.writer.add_figure(
                f"{figure_label}-regionally_ordered", regionally_ordered_figure)
            self.writer.add_figure(f"{figure_label}-regions", regions_figure)
            self.writer.add_figure(
                f"{figure_label}-regions_ordered", regions_ordered_figure)
        else:
            self.writer.add_figure(f"{figure_label}-unordered", figure, index)
            self.writer.add_figure(
                f"{figure_label}-ordered", ordered_figure, index)
            self.writer.add_figure(
                f"{figure_label}-regionally_ordered", regionally_ordered_figure, index)
            self.writer.add_figure(
                f"{figure_label}-regions", regions_figure, index)
            self.writer.add_figure(
                f"{figure_label}-regions_ordered", regions_ordered_figure, index)
        self.writer.flush()
        plt.close("all")
        return

    def calculate_weighted_loss(self, regional_loss_mean, country_loss_mean):
        loss = torch.tensor([], dtype=torch.float32, device=self.device)
        loss.requires_grad = True
        total_loss = (self.regional_portion * regional_loss_mean) + \
            ((1-self.regional_portion) * country_loss_mean)

        loss = torch.cat((loss, total_loss.unsqueeze(0)), dim=0)

        self.writer.add_scalar('Batch Loss', loss.item(), self.batch_count)
        self.writer.add_scalar(
            'Unweighted Regional Batch Loss', regional_loss_mean.item(), self.batch_count)
        self.writer.add_scalar(
            'Unweighted Country Batch Loss', country_loss_mean.item(), self.batch_count)
        self.writer.add_scalar('Weighted Regional Batch Loss',
                               self.regional_portion * regional_loss_mean.item(), self.batch_count)
        self.writer.add_scalar('Weighted Country Batch Loss', (
            1-self.regional_portion) * country_loss_mean.item(), self.batch_count)
        return loss


def create_and_train_model(REPO_PATH: str, seed: int = 1234):
    """
    Creates and trains a model using the specified repository path.

    Args:
        REPO_PATH (str): The path to the repository.

    Returns:
        None
    """

    country_list = f'{REPO_PATH}/data_finding/country_list_region.csv'
    region_list = f'{REPO_PATH}/data_finding/UNSD_Methodology.csv'

    testing_directory = f'{REPO_PATH}/Embeddings/Testing'
    test_df = pd.read_csv(f'{testing_directory}/test_data.csv')
    test_dataset = load_dataset.EmbeddingDataset_from_df(
        test_df, "test")
    test_loader = DataLoader(test_dataset, shuffle=False)

    training_datasets = [
        'geo_weakly_balanced.csv',
        'geo_unbalanced.csv',
        'geo_strongly_balanced.csv',
        'mixed_weakly_balanced.csv',
        'mixed_strongly_balanced.csv'
    ]

    for elem in training_datasets:
        train_df = pd.read_csv(
            f'{REPO_PATH}/Embeddings/Training/{elem}')

        hyperparameters = [
            {'starting_regional_loss_portion': 0.0,
             'regional_loss_decline': 1.0},
            {'starting_regional_loss_portion': 0.25,
             'regional_loss_decline': 1.0},
            {'starting_regional_loss_portion': 0.8,
             'regional_loss_decline': 0.75},
             {'starting_regional_loss_portion': 0.5,
             'regional_loss_decline': 1.0}
        ]
        bs = 260
        if elem == 'geo_strongly_balanced.csv' or elem == 'mixed_strongly_balanced.csv':
            bs = 110
        for i in range(0, len(hyperparameters)):
            model = nn.FinetunedClip()
            trained_model = ModelTrainer(model, train_df, country_list, region_list,
                                         batch_size=bs, num_epochs=1,
                                         starting_regional_loss_portion=hyperparameters[
                                             i]['starting_regional_loss_portion'],
                                         regional_loss_decline=hyperparameters[i]['regional_loss_decline'],
                                         train_dataset_name=elem, seed=seed)
            trained_model.test_model(test_dataset)
    print("END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrained Model')
    parser.add_argument('--user', metavar='str', required=True,
                        help='The user of the gpml group')
    parser.add_argument('--yaml_path', metavar='str', required=True,
                        help='The path to the yaml file with the stored paths')
    # parser.add_argument('--training_dataset_name', metavar='str', required=True, help='the name of the dataset')
    # parser.add_argument('--starting_regional_loss_portion', metavar='float', required=True, help='the starting regional loss portion')
    # parser.add_argument('--regional_loss_decline', metavar='float', required=True, help='the factor with which the regional loss portion is multiplied each epoch')
    args = parser.parse_args()

    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        REPO_PATH = paths['repo_path'][args.user]
        create_and_train_model(REPO_PATH)
