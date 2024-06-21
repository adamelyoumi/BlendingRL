import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CustomSoftmax(nn.Module):
    def __init__(self, M):
        super(CustomSoftmax, self).__init__()
        self.M = M

    def forward(self, x):
        exp_x = th.exp(self.M * x)
        softmax_x = exp_x / exp_x.sum(dim=1, keepdim=True)
        return softmax_x

class CustomNet(nn.Module):
    def __init__(self, input_dim, output_dim, M):
        super(CustomNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.custom_softmax = CustomSoftmax(M)

    def forward(self, x):
        fc_output = self.fc(x)
        relu_out = nn.ReLU()(fc_output)
        softmax_output = self.custom_softmax(relu_out)
        final_output = relu_out * softmax_output
        print(relu_out, softmax_output, final_output, sep="\n\n")
        return final_output

# Example usage:
input_dim = 10  # example input dimension
output_dim = 5  # example output dimension
M = 30  # example value for M

model = CustomNet(input_dim, output_dim, M)

# Generate random input
input_data = th.randn(3, input_dim)  # batch size of 3

# Forward pass
output = model(input_data)