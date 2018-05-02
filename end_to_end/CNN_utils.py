import numpy as np
import torch
from torch.autograd import Variable
import sklearn.metrics

def split_sizes_site(sites):
    """Gets the split sizes to split dataset by site for a dataset with multiple sites.
    Assumes that site-days for with the same site are in blocks of rows in dataset
    
    Arguments:
        sites (array): array indicating the site of each row 
    """
    split_sizes = []
    for i in range(len(sites)):
        if i == 0:
            site = sites[i]
            split_sizes.append(i)
        elif site != sites[i]:
            site = sites[i]
            split_sizes.append(i - (len(split_sizes)-1)*split_sizes[len(split_sizes)-1])
        elif i == len(sites)-1:
            split_sizes.append((i+1) - (len(split_sizes)-1)*split_sizes[len(split_sizes)-1])
    
    split_sizes = split_sizes[1:]
    return split_sizes


def split_data(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0]+split_sizes), dim=0)[:-1]
    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
                 for start, length in zip(splits, split_sizes))


def pad_stack_splits(site_tuple, split_sizes, x_or_y):
    """Zero (x) or nan (y) pads site sequences and stacks them into a matrix.
    
    Arguments:
        site_tuple (tuple): tuple of site sequences to pad and stack
        split_sizes (array): lengths of site sequences
        x_or_y (string): 'x' or 'y' indicating whether to pad and stack x or y
    """
    data_padded_list = []
    for sequence in site_tuple:
        max_sequence_length = torch.max(torch.from_numpy(split_sizes))

        if x_or_y == 'x':
            zero_padding_rows = torch.zeros(max_sequence_length-sequence.size()[0], sequence.size()[1])
            data_padded_list.append(torch.cat((sequence, zero_padding_rows), dim=0))
            
        elif x_or_y == 'y':
            nan_padding = torch.zeros(max_sequence_length - sequence.size()[0]).double()*np.nan
            data_padded_list.append(torch.cat((sequence, nan_padding), dim=0))
            
    return torch.stack(data_padded_list, dim=0)


def get_monitorData_indices(sequence):
    """Gets indices for a site sequence for which there is an output for MonitorData.
    
    Arguments:
        sequence (Tensor): sequence of MonitorData outputs for a given site, including NaNs
    """
    response_indicator_vec = sequence == sequence
    num_responses = torch.sum(response_indicator_vec)
    response_indices = torch.sort(response_indicator_vec, dim=0, descending=True)[1][:num_responses]
    ordered_response_indices = torch.sort(response_indices)[0]
    return ordered_response_indices


def r2(model, batch_size, x_stack_nonConst, x_tuple, y_tuple, get_pred=False):
    """Computes R-squared
    
    Arguments:
        model (torch): model to test
        batch_size (int): to determine how many sequences to read in at a time
        x_stack (tensor): stack of site sequences
        y_tuple (tuple): tuple of true y values by sequence, including NaNs
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
        
        # get indices for which there is monitor data, associated monitor data values and inputs
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
        pred_batch = list(model(x_stack_batch_nonConst, x_batch, y_ind_by_site).data.numpy())
        
        # concatenate new predictions with ones from previous batches
        y += y_batch
        pred += pred_batch
    
    if get_pred == False:
        return sklearn.metrics.r2_score(y, pred)
    elif get_pred == True:
        return sklearn.metrics.r2_score(y, pred), pred


def get_nonConst_vars(data, site_var_name='site', y_var_name='MonitorData', cutoff=1000):
    """Get column names for variables that are not constant within a site sequence
    
    Arguments:
        data (pandas.DataFrame): For checking if variables are non-constant
        site_var_name (str): Column name for monitor site
        y_var_name (str): Column name for monitor output
        cutoff (int): Number of unique values a variable needs to have to be considered non-constant
    """
    site = data.loc[:, site_var_name].values[0]
    data_site = data[data[site_var_name] == site]
    data_site_x = data_site.drop([site_var_name, y_var_name], axis=1)
    
    nonConst_colNames = []
    for i in range(data_site_x.shape[1]):
        if len(np.unique(data_site_x.iloc[:, i].values)) >= cutoff:
            nonConst_colNames.append(data_site_x.columns[i])
    
    return nonConst_colNames


def train_CNN(train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, cnn, optimizer, loss, num_epochs, batch_size):
    
    # get number of batches
    if train_x_std_stack_nonConst.size()[0] % batch_size != 0:
        num_batches = int(np.floor(train_x_std_stack_nonConst.size()[0]/batch_size) + 1)
    else:
        num_batches = int(train_x_std_stack_nonConst.size()[0]/batch_size)

    for epoch in range(num_epochs):
        # get set of shuffled batches for epoch
        batches = np.random.choice(np.arange(train_x_std_stack_nonConst.size()[0]), size=(num_batches-1, batch_size), replace=False)
        epoch_loss = 0

        for batch in batches:  
            
            x_stack_batch_nonConst = train_x_std_stack_nonConst[torch.from_numpy(np.array(batch)).long()]

            # get indices for monitor data and actual monitor data
            y_by_site = []
            x_by_site = []
            y_ind_by_site = []
            for i in range(len(batch)):
                y_ind = get_monitorData_indices(train_y_tuple[batch[i]])
                y_by_site.append(train_y_tuple[batch[i]][y_ind])
                y_ind_by_site.append(y_ind)
                x_by_site.append(train_x_std_tuple[batch[i]][y_ind])
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

        if epoch+1 % 25 == 0:
            print('Epoch loss after epoch ' + str(epoch+1) + ': ' + str(epoch_loss))
            print('Train R^2 after epoch ' + str(epoch+1) + ': ' + str(r2(cnn, batch_size, train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, get_pred=False)))
    
    return None