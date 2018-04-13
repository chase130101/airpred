import torch

### CNN model architecture
class CNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_static, kernel_size, padding):
        super(CNN, self).__init__()
        
        self.conv1d = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding, bias=bias)
        self.norm1 = torch.nn.BatchNorm1d(num_features = hidden_size)
        self.tanh = torch.nn.Tanh()
        #self.norm2 = torch.nn.BatchNorm1d(num_features = hidden_size + num_static)
        self.linear = torch.nn.Linear(in_features = hidden_size + num_static, out_features = 1, bias = True)
        
    def forward(self, input_vary, input_static, y_ind_by_site):
        hidden = self.conv1d(input_vary)
        hidden = self.norm1(hidden)
        hidden = self.tanh(hidden)
        
        hidden_w_response = []
        
        for i in range(hidden.size()[0]):
            hidden_w_response.append(torch.transpose(hidden[i][:, y_ind_by_site[i]], 0, 1)) 
        hidden_w_response = torch.cat(hidden_w_response, dim = 0)
        
        hidden_w_response__input_static = torch.cat([hidden_w_response, input_static], dim = 1)
        
        #hidden = self.norm2(hidden_w_response__input_static)
        output = self.linear(hidden_w_response__input_static)

        return output