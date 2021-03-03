import torch

#--------------------------Useful functions and OPS ---------------------------#
# Functions
def _hashmap(x):
    # List (key, value, key, value, ....)
    return_x = {}
    if len(x)%2 == 0:
        for i in range(0, len(x)-1, 2):
            return_x[x[i]] = torch.tensor(x[i+1])
    else:
        raise IndexError('Un-even key-value pairs')

    return return_x

def _vector(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    else:
        # Maybe single value
        return torch.tensor([x], dtype=torch.float32)

def _put(x, idx_or_key, value):
    if isinstance(x, dict):
        try:
            if not torch.is_tensor(value):
                value = torch.tensor(value)
            x[idx_or_key] = value
        except:
            raise IndexError('Key {} cannot put in the dict'.format(idx_or_key))
        return x
    elif isinstance(x, list):
        try:
            x[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x
    elif torch.is_tensor(x):
        try:
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=x.dtype)
            x[idx_or_key] = value
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))

        return x
    else:
         raise AssertionError('Unsupported data structure')

def _remove(x, idx_or_key):
    if isinstance(x, dict):
        try:
            if isinstance(idx_or_key, float):
                idx_or_key = int(idx_or_key)
            x.pop(idx_or_key, None)
        except:
            raise IndexError('Key {} is not present in the dict'.format(idx_or_key))
        return x
    elif isinstance(x, list):
        try:
            if isinstance(idx_or_key, float):
                idx_or_key = int(idx_or_key)
            x.pop(idx_or_key)
        except:
            raise IndexError('Index {} is not present in the list'.format(idx_or_key))
        return x
    elif torch.is_tensor(x):
        try:
            x = torch.cat((x[:idx_or_key], x[(idx_or_key+1):]))
        except:
            raise IndexError('Index {} is not present in the tensor'.format(idx_or_key))

        return x
    else:
         raise AssertionError('Unsupported data structure')

def _append(x, value):
    if isinstance(x, list):
        if isinstance(value, list):
            x.extend(value)
        else:
            # single value
            x.append(value)
    elif torch.is_tensor(x):
        if not torch.is_tensor(value):
            if isinstance(value, list):
                value = torch.tensor(value, dtype=x.dtype)
            else:
                value = torch.tensor([value], dtype=x.dtype)
        try:
            x = torch.cat((x, value))
        except:
            raise AssertionError('Cannot append the torch tensors')
        return x
    else:
        raise AssertionError('Unsupported data structure')

def _get(x, idx):
    if isinstance(x, dict):
        # Don't change the hash-key
        return x[idx]
    elif isinstance(x, list):
        if isinstance(idx, float):
            idx = int(idx)
        return x[idx]
    elif torch.is_tensor(x):
        try:
            if idx.type() == 'torch.FloatTensor':
                idx = idx.type(torch.LongTensor)
        except:
            idx = torch.tensor(idx, dtype=torch.long)
        return x[idx]
    else:
        raise AssertionError('Unsupported data structure')

def _squareroot(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(x)

def _totensor(x, dtype=torch.float32):
    if not torch.is_tensor(x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=dtype)
        else:
            x = torch.tensor([x], dtype=dtype)
    return x

def _mat_repmat(x, r1, r2):
    # Check if tensor
    if not torch.is_tensor(x):
        raise AssertionError('Cannot REPMAT, because input is not tensor')
    # Check for 0 dimensional tensor
    if torch.is_tensor(r1):
        if r1.shape == torch.Size([]):
            r1 = int(r1.item())
    if torch.is_tensor(r2):
        if r2.shape == torch.Size([]):
            r2 = int(r2.item())
    x = x.repeat(r1, r2)

    return x

def _mat_transpose(x):
    # Smart transpose
    # Check if tensor
    if not torch.is_tensor(x):
        raise AssertionError('Cannot Transpose, because input is not tensor')
    if len(x.shape) == 1:
        # add an extra dimension
        x = x.unsqueeze(1)

    return x.T
