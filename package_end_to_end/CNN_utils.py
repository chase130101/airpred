"""Description: This is a set of functions used for the purposes of training and evaluating a CNN.
Several of these functions are used in data pre-processing, and the last two are evaluation
and training functions. The other scripts in the repository and our tutorial will give the user more 
insight into how all of these functions can be used.
"""
import numpy as np
import torch
from torch.autograd import Variable
import sklearn.metrics

def split_sizes_site(sites):
    """Returns the split sizes to when splitting dataset by site for a dataset with multiple sites
    - Assumes that site-days for the same site are arranged in blocks of rows in the dataset
    - The output from this function can be used as an argument for the split_data
    - The output from this function as a numpy.array can be used as an argument for 
    the pad_stack_splits function below
    -----------
    Inputs:
        - sites (numpy.array or list): Indicates the site of each row in the dataset
    -----------
    Outputs:
        - split_sizes (list of int): Contains the split sizes
    """
    split_sizes = []
    for i in range(len(sites)):
        if i == 0:
            site = sites[i]
            split_sizes.append(i)
        elif site != sites[i]:
            site = sites[i]
            split_sizes.append(i-(len(split_sizes)-1)*split_sizes[len(split_sizes)-1])
        elif i == len(sites)-1:
            split_sizes.append((i+1)-(len(split_sizes)-1)*split_sizes[len(split_sizes)-1])
    
    split_sizes = split_sizes[1:]
    return split_sizes


def split_data(data, split_sizes, dim=0):
    """Splits a dataset into blocks with the sizes indicated by split sizes
    - The output from the split_sizes_site function above can be used as an argument
    for this function
    - The output from this function can be used as an argument for the pad_stack_splits
    function, r2, and train_CNN functions below
    -----------
    Inputs:
        - data (torch.Tensor): Dataset to split into blocks
        - split_sizes (list of int): Sizes of blocks; sum of the sizes cannot exceed the length of the tensor
        along the dimension that it is being split
        - dim (int): Dimension along which to split data
    -----------
    Outputs:
        - (tuple of torch.Tensor): Desired blocks of dataset contained within a tuple
    """
    if dim < 0:
        dim += data.dim()
    
    dim_size = data.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0]+split_sizes), dim=0)[:-1]
    return tuple(data.narrow(int(dim), int(start), int(length)) 
                 for start, length in zip(splits, split_sizes))


def pad_stack_splits(site_tuple, split_sizes, x_or_y):
    """Zero-pads site sequences containing features (x) or nan-pads site sequences containing
    the response (y) and stacks them into a matrix
    - The output from the split_sizes_site as a numpy.array and split_data functions above can 
    be used as arguments for this function
    - If zero-padding features of site sequences, a transpose of the output as a torch.autograd.Variable
    can be used as an argument for the r2 and train_CNN functions below (see tutorial)
    -----------
    Inputs:
        - site_tuple (tuple of torch.Tensor): Blocks of site sequences to pad and stack
        - split_sizes (numpy.array of int): Lengths of site sequences
        -- Length of site_tuple must equal length of split_sizes
        - x_or_y (str): Must be one of 'x' or 'y'; indicates whether to pad and stack 
        features or response values
    -----------
    Outputs:
        - (torch.Tensor): Matrix of padded features or response values of site sequences
    """
    data_padded_list = []
    for sequence in site_tuple:
        max_sequence_length = torch.max(torch.from_numpy(split_sizes))

        if x_or_y == 'x':
            zero_padding_rows = torch.zeros(max_sequence_length-sequence.size()[0], sequence.size()[1])
            data_padded_list.append(torch.cat((sequence, zero_padding_rows), dim=0))
            
        elif x_or_y == 'y':
            nan_padding = torch.zeros(max_sequence_length-sequence.size()[0]).double()*np.nan
            data_padded_list.append(torch.cat((sequence, nan_padding), dim=0))
            
    return torch.stack(data_padded_list, dim=0)


def get_monitorData_indices(sequence):
    """Returns indices of a site sequence for which the response value is non-missing
    - This function is used within the r2 and train_CNN functions below
    -----------
    Inputs:
        - sequence (torch.Tensor): Sequence of response values for a given site, including nans
    -----------
    Outputs:
        - ordered_response_indices (torch.LongTensor): Sequence of indices for which there is a response
        value for a given site
    """
    response_indicator_vec = sequence == sequence
    num_responses = torch.sum(response_indicator_vec)
    response_indices = torch.sort(response_indicator_vec, dim=0, descending=True)[1][:num_responses]
    ordered_response_indices = torch.sort(response_indices)[0]
    return ordered_response_indices


def get_nonConst_vars(data, site_var_name='site', y_var_name='MonitorData', cutoff=1000):
    """Gets column names for variables that are not constant throughout site sequence
    - Associated feature values can be used in the convolutional layer of a CNN
    -----------
    Inputs:
        - data (pandas.DataFrame): For checking if variables are non-constant
        - site_var_name (str): Column name for monitor site
        - y_var_name (str): Column name for monitor output
        - cutoff (int): Number of unique values a variable needs to have to be considered non-constant
    -----------
    Outputs:
        - nonConst_colNames (list): Names of non-constant variables
    """
    site = data.loc[:, site_var_name].values[0]
    data_site = data[data[site_var_name] == site]
    data_site_x = data_site.drop([site_var_name, y_var_name], axis=1)
    
    nonConst_colNames = []
    for i in range(data_site_x.shape[1]):
        if len(np.unique(data_site_x.iloc[:, i].values)) >= cutoff:
            nonConst_colNames.append(data_site_x.columns[i])
    
    return nonConst_colNames


def r2(cnn, batch_size, x_stack_nonConst, x_tuple, y_tuple, get_pred=False):
    """Returns R^2 for predictions of CNN on a given dataset
    - Will return predictions if get_pred is set to True; will only return
    predictions for which there is a corresponding non-missing response value
    - The output from split_data and transposed output from pad_stack_splits as a
    torch.autograd.Variable can be used as arguments for this function
    - This function is used within the train_CNN function below
    -----------
    Inputs:
        - cnn (torch model): CNN used to make predictions
        - batch_size (int): To determine how many site sequences to read in at a time; must
        be less than the number of unique sites in the data that cnn is being evaluated on
        (length of x_tuple or y_tuple)
        - x_stack_nonConst (torch.autograd.Variable): 3D matrix of feature vectors for features
        that change on a daily basis for each site sequence; each 2D matrix along the 0th dimension
        of x_stack_nonConst should be the set of non-constant feature vectors for a site sequence
        in chronological order along the 1st dimension; to be used as input to convolutional 
        layer(s) of cnn
        - x_tuple (tuple of torch.Tensor): Each value in the tuple should be a 2D matrix of feature vectors 
        for a site sequence in chronological order along the 0th dimension
        - y_tuple (tuple of torch.Tensor): True response values by sequence, including nans
        -- Each vector in y_tuple must contain at least one non-missing response value
        -- Length of x_tuple, length of y_tuple, and 0th dimension of x_stack_nonConst must be equal
        -- 0th dimension of each 2D matrix in x_tuple, length of each vector in y_tuple, and 2nd dimension of
        x_stack_nonConst must be equal
        -- 2D matrices along 1st dimension of x_stack_nonConst, matrices in x_tuple, and vectors in y_tuple
        should correspond to the same site sequences
        - get_pred (boolean): Default is false; set to True if predictions are desired
    -----------
    Outputs:
        - (float): R^2 value
        - (list): Only if get_pred set to True; predictions for which there is a corresponding 
        non-missing response
    """
    y = []
    pred = []
    
    # get number of batches
    if x_stack_nonConst.size()[0] % batch_size != 0:
        num_batches = int(np.floor(x_stack_nonConst.size()[0]/batch_size)+1)
    else:
        num_batches = int(x_stack_nonConst.size()[0]/batch_size)
        
    for batch in range(num_batches):
        # get x and y for this batch
        x_stack_batch_nonConst = x_stack_nonConst[batch_size*batch:batch_size*(batch+1)]
        x_tuple_batch = x_tuple[batch_size*batch:batch_size*(batch+1)]
        y_tuple_nans = y_tuple[batch_size*batch:batch_size*(batch+1)]
        
        # get indices for which there is monitor data and associated monitor data values and model inputs
        y_by_site = []
        x_by_site = []
        y_ind_by_site = []
        for i in range(len(y_tuple_nans)):
            y_ind = get_monitorData_indices(y_tuple_nans[i])
            y_by_site.append(y_tuple_nans[i][y_ind])
            y_ind_by_site.append(y_ind)
            x_by_site.append(x_tuple_batch[i][y_ind])
        y_batch = list(Variable(torch.cat(y_by_site, dim=0)).data.numpy())
        x_batch = Variable(torch.cat(x_by_site, dim=0)).float()
        
        # get model output
        pred_batch = list(cnn(x_stack_batch_nonConst, x_batch, y_ind_by_site).data.numpy())
        
        # concatenate new predictions with ones from previous batches
        y += y_batch
        pred += pred_batch
    
    if get_pred == False:
        return sklearn.metrics.r2_score(y, pred)
    elif get_pred == True:
        return sklearn.metrics.r2_score(y, pred), [pred[i][0] for i in range(len(pred))]


def train_CNN(x_stack_nonConst, x_tuple, y_tuple, cnn, optimizer, loss, num_epochs, batch_size):
    """Trains CNN on a dataset
    - The output from split_data and transposed output from pad_stack_splits as a
    torch.autograd.Variable can be used as arguments for this function
    -----------
    Inputs:
        - x_stack_nonConst (torch.autograd.Variable): 3D matrix of feature vectors for features
        that change on a daily basis for each site sequence; each 2D matrix along the 0th dimension
        of x_stack_nonConst should be the set of non-constant feature vectors for a site sequence
        in chronological order along the 1st dimension; to be used as input to convolutional 
        layer(s) of cnn
        - x_tuple (tuple of torch.Tensor): Each value in the tuple should be a 2D matrix of feature vectors 
        for a site sequence in chronological order along the 0th dimension
        - y_tuple (tuple of torch.Tensor): True response values by sequence, including nans
        -- Each vector in y_tuple must contain at least one non-missing response value
        -- Length of x_tuple, length of y_tuple, and 0th dimension of x_stack_nonConst must be equal
        -- 0th dimension of each 2D matrix in x_tuple, length of each vector in y_tuple, and 2nd dimension of
        x_stack_nonConst must be equal
        -- 2D matrices along 1st dimension of x_stack_nonConst, matrices in x_tuple, and vectors in y_tuple
        should correspond to the same site sequences
        - get_pred (boolean): Default is false; set to True if predictions are desired
        - cnn (torch model): CNN used to make predictions
        - optimizer (torch optimizer): Optimizer used for training cnn
        - loss (torch loss function): Loss function to minimize
        - num_epochs (int): Number of epochs to train cnn for
        - batch_size (int): To determine how many site sequences to read in at a time; must
        be less than the number of unique sites in the data that cnn is being evaluated on
        (length of x_tuple or y_tuple)
    -----------
    Outputs:
        - None
    """
    # get number of batches
    if x_stack_nonConst.size()[0] % batch_size != 0:
        num_batches = int(np.floor(x_stack_nonConst.size()[0]/batch_size) + 1)
    else:
        num_batches = int(x_stack_nonConst.size()[0]/batch_size)

    for epoch in range(num_epochs):
        # get set of shuffled batches for epoch
        batches = np.random.choice(np.arange(x_stack_nonConst.size()[0]), 
                                   size=(num_batches-1, batch_size), 
                                   replace=False)
        epoch_loss = 0
        for batch in batches:  
            # get non-constant model inputs for this batch
            x_stack_batch_nonConst = x_stack_nonConst[torch.from_numpy(np.array(batch)).long()]

            # get indices for which there is monitor data and associated monitor data values and model inputs for this batch
            y_by_site = []
            x_by_site = []
            y_ind_by_site = []
            for i in range(len(batch)):
                y_ind = get_monitorData_indices(y_tuple[batch[i]])
                y_by_site.append(y_tuple[batch[i]][y_ind])
                y_ind_by_site.append(y_ind)
                x_by_site.append(x_tuple[batch[i]][y_ind])
            y_batch = Variable(torch.cat(y_by_site, dim=0)).float()
            x_batch = Variable(torch.cat(x_by_site, dim=0)).float()

            # get model output
            pred_batch = cnn(x_stack_batch_nonConst, x_batch, y_ind_by_site)

            # zero gradient, compute loss, backprop, and update parameters
            optimizer.zero_grad()
            loss_batch = loss(pred_batch, y_batch)
            loss_batch.backward()
            optimizer.step()

            # accumulate loss over epoch
            epoch_loss += loss_batch.data[0]
            
        if (epoch+1) % 25 == 0:
            print('Epoch loss after epoch ' + str(epoch+1) + ': ' + str(epoch_loss))
            print('Train R^2 after epoch ' + str(epoch+1) + ': ' + str(r2(cnn, batch_size, x_stack_nonConst, x_tuple, y_tuple, get_pred=False)))
            print()
    
    return None