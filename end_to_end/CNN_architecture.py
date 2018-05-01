import torch

### CNN1 model architecture
class CNN1(torch.nn.Module):
    def __init__(self, input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, dropout_full, hidden_size_combo, dropout_combo):
        super(CNN1, self).__init__()
        
        # convolutional layer for non-constant input variables within a site sequence; normalize output before ReLU; dropout after ReLU
        self.conv1d = torch.nn.Conv1d(in_channels=input_size_conv, out_channels=hidden_size_conv, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm_conv = torch.nn.BatchNorm1d(num_features=hidden_size_conv)
        self.relu_conv = torch.nn.ReLU()
        
        # linear layer for all input variables; normalize output before ReLU; dropout after ReLU
        self.linear_full = torch.nn.Linear(in_features=input_size_full, out_features=hidden_size_full, bias=True)
        self.norm_full = torch.nn.BatchNorm1d(num_features=hidden_size_full)
        self.relu_full = torch.nn.ReLU()
        self.dropout_full = torch.nn.Dropout(p=dropout_full)
        
        # linear layer for combination of outputs from convolution layer and initial linear layer
        # normalize output before ReLU; dropout after ReLU
        # linear layer for model output
        self.linear1_combo = torch.nn.Linear(in_features=hidden_size_conv+hidden_size_full, out_features=hidden_size_combo, bias=True)
        self.norm_combo = torch.nn.BatchNorm1d(num_features=hidden_size_combo)
        self.relu_combo = torch.nn.ReLU()
        self.dropout_combo = torch.nn.Dropout(p=dropout_combo)
        self.linear2_combo = torch.nn.Linear(in_features=hidden_size_combo, out_features=1, bias=True)
        
    def forward(self, input_conv, input_full, y_ind_by_site):
        hidden_conv = self.conv1d(input_conv)
        hidden_conv = self.norm_conv(hidden_conv)
        hidden_conv = self.relu_conv(hidden_conv)
        
        hidden_full = self.linear_full(input_full)
        hidden_full = self.norm_full(hidden_full)
        hidden_full = self.relu_full(hidden_full)
        hidden_full = self.dropout_full(hidden_full)
        
        # get convolution outputs for which there is a response value
        hidden_conv_w_response = []
        for i in range(hidden_conv.size()[0]):
            hidden_conv_w_response.append(torch.transpose(hidden_conv[i][:, y_ind_by_site[i]], 0, 1)) 
        hidden_conv_w_response = torch.cat(hidden_conv_w_response, dim=0)
        
        # concatenate convolution outputs for which there is a response value with outputs from initial linear layer
        hidden_conv_w_response__hidden_full = torch.cat([hidden_conv_w_response, hidden_full], dim=1)
        hidden_combo = self.linear1_combo(hidden_conv_w_response__hidden_full)
        hidden_combo = self.norm_combo(hidden_combo)
        hidden_combo = self.relu_combo(hidden_combo)
        hidden_combo = self.dropout_combo(hidden_combo)
        output = self.linear2_combo(hidden_combo)

        return output
    
    
class CNN2(torch.nn.Module):
    def __init__(self, input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, dropout_full, hidden_size2_full, dropout2_full):
        super(CNN2, self).__init__()
        
        # convolutional layer for non-constant input variables within a site sequence; normalize output before ReLU; dropout after ReLU
        self.conv1d = torch.nn.Conv1d(in_channels=input_size_conv, out_channels=hidden_size_conv, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm_conv = torch.nn.BatchNorm1d(num_features=hidden_size_conv)
        self.relu_conv = torch.nn.ReLU()
        
        # linear layer for all input variables; normalize output before ReLU; dropout after ReLU
        self.linear_full = torch.nn.Linear(in_features=input_size_full, out_features=hidden_size_full, bias=True)
        self.norm_full = torch.nn.BatchNorm1d(num_features=hidden_size_full)
        self.relu_full = torch.nn.ReLU()
        self.dropout_full = torch.nn.Dropout(p=dropout_full)
        self.linear2_full = torch.nn.Linear(in_features=hidden_size_full, out_features=hidden_size2_full, bias=True)
        self.relu2_full = torch.nn.ReLU()
        self.dropout2_full = torch.nn.Dropout(p=dropout2_full)
        
        # linear layer for combination of outputs from convolution layer and initial linear layer
        # normalize output before ReLU; dropout after ReLU
        # linear layer for model output
        self.linear_combo = torch.nn.Linear(in_features=hidden_size_conv+hidden_size2_full, out_features=1, bias=True)
        
    def forward(self, input_conv, input_full, y_ind_by_site):
        hidden_conv = self.conv1d(input_conv)
        hidden_conv = self.norm_conv(hidden_conv)
        hidden_conv = self.relu_conv(hidden_conv)
        
        hidden_full = self.linear_full(input_full)
        hidden_full = self.norm_full(hidden_full)
        hidden_full = self.relu_full(hidden_full)
        hidden_full = self.dropout_full(hidden_full)
        hidden_full = self.linear2_full(hidden_full)
        hidden_full = self.relu2_full(hidden_full)
        hidden_full = self.dropout2_full(hidden_full)
        
        # get convolution outputs for which there is a response value
        hidden_conv_w_response = []
        for i in range(hidden_conv.size()[0]):
            hidden_conv_w_response.append(torch.transpose(hidden_conv[i][:, y_ind_by_site[i]], 0, 1)) 
        hidden_conv_w_response = torch.cat(hidden_conv_w_response, dim=0)
        
        # concatenate convolution outputs for which there is a response value with outputs from initial linear layer
        hidden_conv_w_response__hidden_full = torch.cat([hidden_conv_w_response, hidden_full], dim=1)
        output = self.linear_combo(hidden_conv_w_response__hidden_full)

        return output