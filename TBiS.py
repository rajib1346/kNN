import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
# Using only the first feature as an example
dataset = data.data[:, 0]

def bitonic_sort(arr, up=True):
    """Basic Bitonic Sort implementation."""
    if len(arr) <= 1:
        return arr

    # Splitting the input array
    first = bitonic_sort(arr[:len(arr) // 2], True)
    second = bitonic_sort(arr[len(arr) // 2:], False)

    merged = merge(first + second, up)
    return merged

def merge(arr, up):
    """Merging for Bitonic Sort."""
    if len(arr) == 1:
        return arr
    bitonic_compare(arr, up)
    first = merge(arr[:len(arr) // 2], up)
    second = merge(arr[len(arr) // 2:], up)
    return first + second

def bitonic_compare(arr, up):
    """Compare and swap operation for Bitonic Sort."""
    dist = len(arr) // 2
    for i in range(dist):
        if (arr[i] > arr[i + dist]) == up:
            arr[i], arr[i + dist] = arr[i + dist], arr[i]

def truncated_bitonic_search(arr, k):
    """TBiS - Truncated Bitonic Sort to find k smallest elements."""
    sorted_arr = bitonic_sort(arr)
    return sorted_arr[:k]

# Demonstration:
k = 10
result = truncated_bitonic_search(dataset, k)
print(result)
