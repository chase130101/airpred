"""Description: These are the two CNN architectures that we fitted and evaluated on the air pollution dataset.
There is only a subset of features that changes daily over the course of a site sequence (referred to as non-constant 
features). The other features stay constant over the course of a site sequence. We feed the subset of features that change 
on a daily basis to a convolutional layer followed by normalization and ReLU and all of the features (or features of the
user's choosing) to one (CNN1) or two (CNN2) linear layers with normalization, ReLU activation functions, and dropout. 
The convolutional layer should help to account for the relationship between features of consecutive days when producing 
air pollution predictions. The hidden units following the linear layer(s) and convolutional layer are concatenated 
and then fed to another linear layer followed by normalization, ReLU, dropout, and a linear layer the produces the model
output (CNN1) or a linear layer that produces the model output (CNN2). The 2 sets of hidden units that are concatenated 
correspond to the same site-day.

The explicit usage of these architectures in our scripts beyond instantiation is abstracted away from the user for training 
and evaluation. The user can see how they are being used in the train_CNN and r2 functions in CNN_utils.py. 

We include diagrams of the CNN architectures on our website for clarity. The usage of the predict method can be seen in our
tutorial.
"""
import torch

### CNN1 model architecture
class CNN1(torch.nn.Module):
    """This class specifies a particular CNN model architecture that can be fitted and then used to predict
    -----------
    Methods:
        - forward: Maps model inputs to outputs during training only for inputs for which the corresponding 
        response value is non-missing
        -- Should never be called explicitly (this is an odd PyTorch thing)
        -- Is called when this line is run following instantiation: CNN1(input_conv, input_full, y_ind_by_site)
        - predict: Maps model inputs to outputs for all inputs
    -----------
    Arguments user needs to specify when instantiating class:
        - input_size_conv (int): Number of feautures that convolutional layer takes as inputs
        - hidden_size_conv (int): Number of hidden units associated with each time point for inputted features from a full
        site sequence to the convolutional layer
        - kernel_size (int): Kernel size of convolutional layer; should be an odd number > 1; if 3, then for time point t, 
        features from t-1, t, and t+1 will be used to produce outputs from convolutional layer for time t; if 5, then for 
        time point t, features from t-2, t-1, t, t+1, t+2 will be used to produce outputs from convolutional layer for time t
        - padding (int): Zero padding for beginning and end of site sequence; should be 1 if kernel_size is set to 3, 2 if
        kernel_size is set to 5, etc.
        - input_size_full (int): Number of features that initial linear layer takes as inputs
        - hidden_size_full (int): Number of hidden units associated with inputted features to initial linear layer
        - dropout_full (float): Dropout probability of hidden units associated with hidden_size_full; must be between 
        0 and 1, inclusive
        - hidden_size_combo (int): Number of hidden units following linear layer that takes the concatenation of hidden units
        associated with hidden_size_conv and hidden_size_full
        - dropout_combo (float): Dropout probability of hidden units associated with hidden_size_combo; must be between 
        0 and 1, inclusive
    -----------
    Attributes 
        See __init__: The attributes are the convolution layer, linear layers, normalization layers, and dropout layers and are
        of type torch.nn.Module
    """
    def __init__(self, input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, dropout_full, hidden_size_combo, dropout_combo):
        super(CNN1, self).__init__()
        
        # convolutional layer to input non-constant variables within a site sequence; normalize output before ReLU; dropout after ReLU
        self.conv1d = torch.nn.Conv1d(in_channels=input_size_conv, out_channels=hidden_size_conv, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm_conv = torch.nn.BatchNorm1d(num_features=hidden_size_conv)
        self.relu_conv = torch.nn.ReLU()
        
        # linear layer to input all variables; normalize output before ReLU; dropout after ReLU
        self.linear_full = torch.nn.Linear(in_features=input_size_full, out_features=hidden_size_full, bias=True)
        self.norm_full = torch.nn.BatchNorm1d(num_features=hidden_size_full)
        self.relu_full = torch.nn.ReLU()
        self.dropout_full = torch.nn.Dropout(p=dropout_full)
        
        # linear layer for combination of outputs from convolution layer and initial linear layer/ReLU
        # normalize output before ReLU; dropout after ReLU
        # linear layer for model output
        self.linear1_combo = torch.nn.Linear(in_features=hidden_size_conv+hidden_size_full, out_features=hidden_size_combo, bias=True)
        self.norm_combo = torch.nn.BatchNorm1d(num_features=hidden_size_combo)
        self.relu_combo = torch.nn.ReLU()
        self.dropout_combo = torch.nn.Dropout(p=dropout_combo)
        self.linear2_combo = torch.nn.Linear(in_features=hidden_size_combo, out_features=1, bias=True)
        
    def forward(self, input_conv, input_full, y_ind_by_site):
        """Maps model inputs to outputs during training/evaluation; only for maps inputs to outputs for which 
        the corresponding response value is non-missing
        - Should never be called explicitly (this is an odd PyTorch thing)
        - Is called when this line is run following instantiation: CNN1(input_conv, input_full, y_ind_by_site)
        - The documentation for this method is going to be hard to follow out of context; this method is implicitly 
        called within the train_CNN function and r2 function of CNN_utils.py, so its usage is abstracted away from
        the user
        -----------
        Inputs:
            - input_conv (torch.autograd.Variable): 3D matrix of feature vectors for features
            that change on a daily basis for each site sequence; each 2D matrix along the 0th dimension
            of input_conv should be the set of non-constant feature vectors for a site sequence
            in chronological order along the 1st dimension; to be used as input to convolutional 
            layer
            - input_full (torch.autograd.Variable): 2D matrix of feature vectors where each row
            is a set of features associated with a site-day for which there is a response value
            -- ith row in input_full should correspond to the ith value in the concatenation of non-missing 
            response values read in as part of a batch of site sequences for training/evaluation
            - y_ind_by_site (list of torch.Tensor): Vectors of indices for each site sequence indicating
            non-missing response values; used to index the hidden units associated with the convolutional layer
            for which there is a non-missing response value
            -- When reading in a batch of site sequences for training/evaluation, the ith vector in y_ind_by_site 
            corresponds to the response values of the ith site sequence; the jth value in the ith vector of y_ind_by_site 
            is the index of the jth day in ith site sequence that has a non-missing response value
            -- 2nd dimension of input_conv and length of y_ind_by_site must be equal
            -- 2D matrices along 0th dimension of input_conv and vectors in y_ind_by_site should correspond 
            to the same site sequences
            -- Sum of lengths of vectors in y_ind_by_site should be equal to 0th dimension of input_full
        -----------
        Outputs:
            - output (torch.autograd.Variable): Vector of model outputs for which the corresponding response value is 
            non-missing
            -- Length of output will be equal to the 0th dimension of input_full and the sum of the lengths of vectors
            in y_ind_by_site
        """
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
        
        # concatenate convolution outputs for which there is a response value with outputs from initial linear layer + ReLU
        hidden_conv_w_response__hidden_full = torch.cat([hidden_conv_w_response, hidden_full], dim=1)
        hidden_combo = self.linear1_combo(hidden_conv_w_response__hidden_full)
        hidden_combo = self.norm_combo(hidden_combo)
        hidden_combo = self.relu_combo(hidden_combo)
        hidden_combo = self.dropout_combo(hidden_combo)
        output = self.linear2_combo(hidden_combo)

        return output
    
    def predict(self, input_conv, input_full):
        """Maps model inputs to outputs for all inputs
        -----------
        Inputs:
            - input_conv (torch.autograd.Variable): 3D matrix of feature vectors for features
            that change on a daily basis for each site sequence; each 2D matrix along the 0th dimension
            of input_conv should be the set of non-constant feature vectors for a site sequence
            in chronological order along the 1st dimension; to be used as input to convolutional 
            layer
            - input_full (torch.autograd.Variable): 2D matrix of feature vectors across all site-days to
            make predictions for
            -- 0th dimension of input_full should be equal to 0th dimension times 2nd dimension of input_conv
        -----------
        Outputs:
            - output (torch.autograd.Variable): Vector of model outputs
            -- Length of output will be equal to the 0th dimension of input_full and 0th dimension times 2nd 
            dimension of input_conv
        """
        hidden_conv = self.conv1d(input_conv)
        hidden_conv = self.norm_conv(hidden_conv)
        hidden_conv = self.relu_conv(hidden_conv)
        hidden_conv = torch.transpose(torch.cat(hidden_conv, dim=1), 0, 1)

        hidden_full = self.linear_full(input_full)
        hidden_full = self.norm_full(hidden_full)
        hidden_full = self.relu_full(hidden_full)
        hidden_full = self.dropout_full(hidden_full)
        
        # concatenate convolution outputs and outputs from initial linear layer + ReLU
        hidden_conv__hidden_full = torch.cat([hidden_conv, hidden_full], dim=1)
        hidden_combo = self.linear1_combo(hidden_conv__hidden_full)
        hidden_combo = self.norm_combo(hidden_combo)
        hidden_combo = self.relu_combo(hidden_combo)
        hidden_combo = self.dropout_combo(hidden_combo)
        output = self.linear2_combo(hidden_combo)
        
        return output

    
### CNN2 model architecture
class CNN2(torch.nn.Module):
    """This class specifies a particular CNN model architecture that can be fitted and then used to predict
    -----------
    Methods:
        - forward: Maps model inputs to outputs during training only for inputs for which the corresponding 
        response value is non-missing
        -- Should never be called explicitly (this is an odd PyTorch thing)
        -- Is called when this line is run following instantiation: CNN1(input_conv, input_full, y_ind_by_site)
        - predict: Maps model inputs to outputs for all inputs
    -----------
    Arguments user needs to specify when instantiating class:
        - input_size_conv (int): Number of feautures that convolutional layer takes as inputs
        - hidden_size_conv (int): Number of hidden units associated with each time point for inputted features from a full
        site sequence to the convolutional layer
        - kernel_size (int): Kernel size of convolutional layer; should be an odd number > 1; if 3, then for time point t, 
        features from t-1, t, and t+1 will be used to produce outputs from convolutional layer for time t; if 5, then for 
        time point t, features from t-2, t-1, t, t+1, t+2 will be used to produce outputs from convolutional layer for time t
        - padding (int): Zero padding for beginning and end of site sequence; should be 1 if kernel_size is set to 3, 2 if
        kernel_size is set to 5, etc.
        - input_size_full (int): Number of features that initial linear layer takes as inputs
        - hidden_size_full (int): Number of hidden units in 1st layer of hidden units associated with inputted features to
        initial linear layer
        - dropout_full (float): Dropout probability of hidden units associated with hidden_size_full; must be between 
        0 and 1, inclusive
        - hidden_size2_full (int): Number of hidden units in 2nd layer of hidden units associated with inputted features to
        initial linear layer
        - dropout2_full (float): Dropout probability of hidden units associated with hidden_size2_full; must be between 
        0 and 1, inclusive
    -----------
    Attributes 
        See __init__: The attributes are the convolution layer, linear layers, normalization layers, and dropout layers and are
        of type torch.nn.Module
    """
    def __init__(self, input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, dropout_full, hidden_size2_full, dropout2_full):
        
        super(CNN2, self).__init__()
        
        # convolutional layer to input non-constant variables within a site sequence; normalize output before ReLU; dropout after ReLU
        self.conv1d = torch.nn.Conv1d(in_channels=input_size_conv, out_channels=hidden_size_conv, kernel_size=kernel_size, padding=padding, bias=True)
        self.norm_conv = torch.nn.BatchNorm1d(num_features=hidden_size_conv)
        self.relu_conv = torch.nn.ReLU()
        
        # 2 fully connected linear layers with first layer taking all input variables; normalize output before ReLU activations; dropout after ReLU activations
        self.linear_full = torch.nn.Linear(in_features=input_size_full, out_features=hidden_size_full, bias=True)
        self.norm_full = torch.nn.BatchNorm1d(num_features=hidden_size_full)
        self.relu_full = torch.nn.ReLU()
        self.dropout_full = torch.nn.Dropout(p=dropout_full)
        self.linear2_full = torch.nn.Linear(in_features=hidden_size_full, out_features=hidden_size2_full, bias=True)
        self.relu2_full = torch.nn.ReLU()
        self.dropout2_full = torch.nn.Dropout(p=dropout2_full)
        
        # linear layer for combination of outputs from convolution layer and fully connected layers
        # produces model output
        self.linear_combo = torch.nn.Linear(in_features=hidden_size_conv+hidden_size2_full, out_features=1, bias=True)
        
    def forward(self, input_conv, input_full, y_ind_by_site):
        """Maps model inputs to outputs during training/evaluation; only for maps inputs to outputs for which 
        the corresponding response value is non-missing
        - Should never be called explicitly (this is an odd PyTorch thing)
        - Is called when this line is run following instantiation: CNN1(input_conv, input_full, y_ind_by_site)
        - The documentation for this method is going to be hard to follow out of context; this method is implicitly 
        called within the train_CNN function and r2 function of CNN_utils.py, so its usage is abstracted away from
        the user
        -----------
        Inputs:
            - input_conv (torch.autograd.Variable): 3D matrix of feature vectors for features
            that change on a daily basis for each site sequence; each 2D matrix along the 0th dimension
            of input_conv should be the set of non-constant feature vectors for a site sequence
            in chronological order along the 1st dimension; to be used as input to convolutional 
            layer
            - input_full (torch.autograd.Variable): 2D matrix of feature vectors where each row
            is a set of features associated with a site-day for which there is a response value
            -- ith row in input_full should correspond to the ith value in the concatenation of non-missing 
            response values read in as part of a batch of site sequences for training/evaluation
            - y_ind_by_site (list of torch.Tensor): Vectors of indices for each site sequence indicating
            non-missing response values; used to index the hidden units associated with the convolutional layer
            for which there is a non-missing response value
            -- When reading in a batch of site sequences for training/evaluation, the ith vector in y_ind_by_site 
            corresponds to the response values of the ith site sequence; the jth value in the ith vector of y_ind_by_site 
            is the index of the jth day in ith site sequence that has a non-missing response value
            -- 2nd dimension of input_conv and length of y_ind_by_site must be equal
            -- 2D matrices along 0th dimension of input_conv and vectors in y_ind_by_site should correspond 
            to the same site sequences
            -- Sum of lengths of vectors in y_ind_by_site should be equal to 0th dimension of input_full
        -----------
        Outputs:
            - output (torch.autograd.Variable): Vector of model outputs for which the corresponding response value is 
            non-missing
            -- Length of output will be equal to the 0th dimension of input_full and the sum of the lengths of vectors
            in y_ind_by_site
        """
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
        
        # concatenate convolution outputs for which there is a response value with outputs from fully connected layers
        hidden_conv_w_response__hidden_full = torch.cat([hidden_conv_w_response, hidden_full], dim=1)
        output = self.linear_combo(hidden_conv_w_response__hidden_full)

        return output
    
    def predict(self, input_conv, input_full):
        """Maps model inputs to outputs for all inputs
        -----------
        Inputs:
            - input_conv (torch.autograd.Variable): 3D matrix of feature vectors for features
            that change on a daily basis for each site sequence; each 2D matrix along the 0th dimension
            of input_conv should be the set of non-constant feature vectors for a site sequence
            in chronological order along the 1st dimension; to be used as input to convolutional 
            layer
            - input_full (torch.autograd.Variable): 2D matrix of feature vectors across all site-days to
            make predictions for
            -- 0th dimension of input_full should be equal to 0th dimension times 2nd dimension of input_conv
        -----------
        Outputs:
            - output (torch.autograd.Variable): Vector of model outputs
            -- Length of output will be equal to the 0th dimension of input_full and 0th dimension times 2nd 
            dimension of input_conv
        """
        hidden_conv = self.conv1d(input_conv)
        hidden_conv = self.norm_conv(hidden_conv)
        hidden_conv = self.relu_conv(hidden_conv)
        hidden_conv = torch.transpose(torch.cat(hidden_conv, dim=1), 0, 1)
        
        hidden_full = self.linear_full(input_full)
        hidden_full = self.norm_full(hidden_full)
        hidden_full = self.relu_full(hidden_full)
        hidden_full = self.dropout_full(hidden_full)
        hidden_full = self.linear2_full(hidden_full)
        hidden_full = self.relu2_full(hidden_full)
        hidden_full = self.dropout2_full(hidden_full)
        
        # concatenate convolution outputs and outputs from fully connected layers
        hidden_conv__hidden_full = torch.cat([hidden_conv, hidden_full], dim=1)
        output = self.linear_combo(hidden_conv__hidden_full)
        
        return output