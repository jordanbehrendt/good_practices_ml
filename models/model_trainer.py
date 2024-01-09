import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from models import nn
import os
sys.path.append('./scripts')
from scripts import load_dataset
import sklearn.model_selection
from torch.utils.data import DataLoader, TensorDataset

class ModelTrainer():

    def __init__(self, model: torch.nn.Module, train_loader, val_loader, num_epochs = 10, learning_rate = 0.001) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = Regional_Loss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.country_list = pd.read_csv("/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data_finding/country_list.csv")
        self.region_list = pd.read_csv("/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/data/UNSD_Methodology.csv")
        self.start_training()


    def start_training(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)  # Calculate the loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                running_loss += loss.item()

            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in self.val_loader:
                    val_outputs = self.model(val_inputs)
                    val_loss += self.criterion(val_outputs, val_targets).item()

            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Train Loss: {running_loss/len(self.train_loader):.4f} - Val Loss: {val_loss/len(self.val_loader):.4f}")

    def test_model(self, test_loader):
        criterion = Regional_Loss()
        test_loss = 0.0
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_outputs = self.model(test_inputs)
                test_loss += criterion(test_outputs, test_targets).item()
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")

class Regional_Loss(torch.nn.Module):
    def __init__(self):
        super(Regional_Loss, self).__init__()

    def forward(self, output, target):
        ref_country = target
        ref_alph2code = self.country_list[self.country_list['Country'] == target]['Alpha2Code'].values[0]
        ref_region = self.region_list[self.region_list['ISO-alpha2 Code'] == ref_alph2code]['Intermediate Region Name'].values[0]
        pred_country = self.country_list['Country'].iloc[np.argmax(np.array(output))]
        pred_alpha2code = self.country_list[self.country_list['Country'] == pred_country]['Alpha2Code'].values[0]
        pred_region = self.region_list[self.region_list['ISO-alpha2 Code'] == pred_alpha2code]['Intermediate Region Name'].values[0]

        if ref_country == pred_country:
            return 0
        elif ref_region == pred_region:
            return 0.5
        else:
            return 1
        

# Directory containing CSV files
directory = '/share/temp/bjordan/good_practices_in_machine_learning/good_practices_ml/Embeddings/Image/'

# Get a list of filenames that start with "geoguessr" and end with ".csv"
file_list = [file for file in os.listdir(directory) if file.startswith('geoguessr') and file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through the files, read them as DataFrames, and append to the list
for file in file_list:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

train, test = sklearn.model_selection.train_test_split(combined_df, test_size=0.2, random_state=1234, shuffle=True)
val, test = sklearn.model_selection.train_test_split(test, test_size=0.5, random_state=1234, shuffle=True)

train_dataset = load_dataset.EmbeddingDataset_from_df(train, "train")
val_dataset = load_dataset.EmbeddingDataset_from_df(val, "val")
test_dataset = load_dataset.EmbeddingDataset_from_df(test, "test")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)



model = nn.FinetunedClip()
trainer = ModelTrainer(model, train_loader, val_loader)
trainer.test_model(test_loader)
print("END")
