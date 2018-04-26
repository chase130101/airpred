import torch

### CNN model architecture
class CNN(torch.nn.Module):
    def __init__(self, input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, hidden_size_combo):
        super(CNN, self).__init__()
        
        self.conv1d = torch.nn.Conv1d(in_channels=input_size_conv, out_channels=hidden_size_conv, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm_conv = torch.nn.BatchNorm1d(num_features = hidden_size_conv)
        self.relu_conv = torch.nn.ReLU()
        
        self.linear_full = torch.nn.Linear(in_features = input_size_full, out_features = hidden_size_full, bias = True)
        self.norm_full = torch.nn.BatchNorm1d(num_features = hidden_size_full)
        self.relu_full = torch.nn.ReLU()
        
        self.linear1_combo = torch.nn.Linear(in_features = hidden_size_conv + hidden_size_full, out_features = hidden_size_combo, bias = True)
        self.norm_combo = torch.nn.BatchNorm1d(num_features = hidden_size_combo)
        self.relu_combo = torch.nn.ReLU()
        self.linear2_combo = torch.nn.Linear(in_features = hidden_size_combo, out_features = 1, bias = True)
        
    def forward(self, input_conv, input_full, y_ind_by_site):
        hidden_conv = self.conv1d(input_conv)
        hidden_conv = self.norm_conv(hidden_conv)
        hidden_conv = self.relu_conv(hidden_conv)
        
        hidden_full = self.linear_full(input_full)
        hidden_full = self.norm_full(hidden_full)
        hidden_full = self.relu_full(hidden_full)
        
        hidden_conv_w_response = []
        for i in range(hidden_conv.size()[0]):
            hidden_conv_w_response.append(torch.transpose(hidden_conv[i][:, y_ind_by_site[i]], 0, 1)) 
        hidden_conv_w_response = torch.cat(hidden_conv_w_response, dim = 0)
                
        hidden_conv_w_response__hidden_full = torch.cat([hidden_conv_w_response, hidden_full], dim = 1)
        hidden_combo = self.linear1_combo(input_full)
        hidden_combo = self.norm_combo(hidden_combo)
        hidden_combo = self.relu_combo(hidden_combo)
        output = self.linear2_combo(hidden_combo)

        return output