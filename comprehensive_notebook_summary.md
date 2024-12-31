# NumPy Library

## Introduction
NumPy (Numerical Python) is a powerful library for numerical computing in Python. It provides support for multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these data structures.

---

## Key Features
- **N-dimensional array object (ndarray)**: Efficient storage and manipulation of large datasets.
- **Mathematical functions**: Supports algebra, trigonometry, statistics, etc.
- **Broadcasting**: Perform operations on arrays of different shapes.
- **Integration with other libraries**: Works seamlessly with libraries like Pandas, Matplotlib, and SciPy.

---

## Installation
```bash
pip install numpy
```

---

## Importing NumPy
```python
import numpy as np
```

---

## Core Concepts

### 1. **ndarray**
The main data structure in NumPy is the `ndarray`, which represents a multi-dimensional array.

### Creating Arrays
```python
# 1D array
arr1 = np.array([1, 2, 3])

# 2D array
arr2 = np.array([[1, 2], [3, 4]])

# 3D array
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

### Array Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Dimensions of the array
print(arr.size)   # Total number of elements
print(arr.dtype)  # Data type of elements
print(arr.ndim)   # Number of dimensions
```

### Array Functions
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(len(arr)) # Number of nested values
print(np.size(arr)) # Number of elements
print(type(arr)) # Data type of variable
print(arr.astype(int)) # Change type
```

### Data Types
NumPy supports a variety of data types such as `int32`, `float64`, `bool`, etc.
```python
arr = np.array([1.5, 2.3, 3.4], dtype='float32')
```

---

## Array Operations

### 1. **Arithmetic Operations**
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 + arr2) # Addition
print(arr1 - arr2)  # Subtraction
print(arr1 * arr2)  # Element-wise multiplication
print(arr1 / arr2)  # Element-wise division
```

### 2. **Universal Functions (ufuncs)**
NumPy provides many mathematical functions.
```python
arr = np.array([1, 4, 9, 16])
print(np.add(arr1, arr2)) # Addition
print(np.subtract(arr1, arr2)) # Subtract
print(np.multiply(arr1, arr2)) # Multiply
print(np.divide(arr1, arr2)) # Divide
print(np.sqrt(arr))  # Square root
print(np.log(arr))  # Natural logarithm
print(np.exp(arr))  # Exponential
print(np.power(arr))  # Power
```

### 3. **Aggregations**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr))  # Sum of all elements
print(np.mean(arr)) # Mean of elements
print(np.max(arr))  # Maximum value
print(np.min(arr))  # Minimum value
print(np.std(arr))  # Standard deviation
print(np.cumsum(arr))  # Cumulative Sum
print(np.cumprod(arr))  # Cumulative Product
```

---

## Array Manipulation

### 1. **Reshaping**
```python
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape(2, 3)
print(reshaped)
```

### 2. **Concatenate**
```python
arr1 = np.array([10, 20, 30])
arr2 = np.array([40, 50, 60])
conct_arr = np.concatenate([arr1, arr2])
print(conct_arr)

arr3 = np.array([[1, 2], [3, 4]])
arr4 = np.array([[5, 6], [7, 8]])
print(np.concatenate([arr3, arr4], axis = 1)) # axis applies on 2d or 3d array

print(np.hstack((arr3, arr4)))  # Horizontal stacking (axis = 0)
print(np.vstack((arr3, arr4)))  # Vertical stacking (axis = 1)
```

### 3. **Flattening**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.flatten())
```

### 4. **Splitting**
```python
arr = np.array([1, 2, 3, 4, 5, 6])
split = np.split(arr, 3)
print(split)
```

---

## Sort, Search and Filter
### 1. **Sort**
```python
arr = np.array([3, 7, 2, 4, 1], [4, 9, 5, 8])
print(np.sort(arr))
```

### 2. **Search**
```python
arr = np.array([3, 7, 2, 4, 1])
print(np.where(arr > 3))

print(np.searchsorted(arr, 2))
```

### 3. **Filter**
```python
arr = np.array([20, 30, 40, 50, 60])
fa = [True, False, True, False, False] # Must be same length as array
print(arr[fa])

print(arr[arr > 40])
```

---

## Random Numbers

### Generating Random Arrays
```python
from numpy.random import default_rng
rng = default_rng()

# Random numbers between 0 and 1
print(rng.random((3, 3)))

# Random integers
print(rng.integers(1, 10, size=(3, 3)))
```

---

## Conclusion
NumPy is a foundational library for scientific computing in Python, offering efficient operations and extensive functionality for array and matrix processing. It is widely used in data analysis, machine learning, and numerical simulations.
