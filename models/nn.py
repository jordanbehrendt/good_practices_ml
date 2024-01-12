import torch

class FinetunedClip(torch.nn.Module):

    def __init__(self, input_size = 212, hidden_size = 212, output_size = 211):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, input_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x