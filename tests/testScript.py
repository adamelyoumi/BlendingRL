import torch

def replace_columns(X, Y, indices):
    # Ensure X and Y have the same batch size
    assert X.shape[0] == Y.shape[0], "Batch sizes of X and Y must match"
    
    batch_size, N = X.shape
    _, M = Y.shape
    
    # Ensure indices length matches the number of columns in Y
    assert len(indices) == M, "Length of indices must match the number of columns in Y"
    
    # Ensure indices are within the range of columns of X
    assert all(0 <= idx < N for idx in indices), "Indices must be within the range of columns of X"

    # Create a copy of X to avoid modifying the original tensor
    X_new = X.clone()
    
    for i, idx in enumerate(indices):
        X_new[:, idx] = Y[:, i]
    
    return X_new

# Example usage:
batch_size = 3
N = 5
M = 2

X = torch.tensor([[10, 20, 30, 40, 50],
                  [60, 70, 80, 90, 100],
                  [110, 120, 130, 140, 150]], dtype=torch.float)

Y = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=torch.float)

indices = [1, 3]

X_new = replace_columns(X, Y, indices)
print(X_new)