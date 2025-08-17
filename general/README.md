# General Python Module

A comprehensive Python module demonstrating standard library usage, container operations, and common algorithms.

## Module Structure

```
general/
├── __init__.py          # Module initialization and exports
├── containers.py        # Container usage examples
├── algorithms.py        # Algorithm implementations
├── stdlib_examples.py   # Standard library demonstrations
└── README.md           # This file
```

## Features

### 📦 Containers (`containers.py`)

- **List Operations**: Append, extend, comprehensions, slicing
- **Dictionary Operations**: CRUD operations, comprehensions, merging
- **Set Operations**: Union, intersection, difference, symmetric difference
- **Tuple Operations**: Basic operations and named tuples
- **Advanced Containers**: defaultdict, Counter, deque, OrderedDict
- **Performance Tips**: Comparison of different container types

### 🧮 Algorithms (`algorithms.py`)

- **Sorting Algorithms**: Bubble sort, quick sort, merge sort, heap sort
- **Searching Algorithms**: Linear search, binary search (iterative & recursive)
- **Dynamic Programming**: Fibonacci, longest common subsequence
- **Graph Algorithms**: Depth-first search (DFS), breadth-first search (BFS)
- **Utility Functions**: Performance benchmarking, random data generation

### 📚 Standard Library (`stdlib_examples.py`)

- **OS Module**: File system operations, environment variables
- **SYS Module**: System information, command-line arguments
- **DateTime Module**: Date/time manipulation, formatting, arithmetic
- **JSON Module**: Serialization and deserialization
- **Regular Expressions**: Pattern matching, substitution, splitting
- **Itertools**: Infinite iterators, combinatorial functions
- **Functools**: Partial functions, reduce, caching decorators
- **Math Module**: Mathematical functions and constants
- **Pathlib**: Modern path manipulation
- **String Module**: String constants and templates
- **Cryptographic Functions**: Hashing, encoding, UUID generation
- **URL Processing**: Parsing and manipulation
- **Random Module**: Random number generation and sampling

## Usage

### Direct Module Import

```python
from general import containers, algorithms, stdlib_examples

# Run all container examples
containers.demonstrate_all_containers()

# Run specific algorithm demos
algorithms.demonstrate_sorting_algorithms()
algorithms.demonstrate_searching_algorithms()

# Run standard library examples
stdlib_examples.demonstrate_all_stdlib()
```

### Using the Demo Script

```bash
# Run all demonstrations
python demo_general.py

# Interactive mode
python demo_general.py --interactive

# Run specific module
python demo_general.py --module containers
python demo_general.py --module algorithms
python demo_general.py --module stdlib
```

### Individual Function Usage

```python
from general.algorithms import quick_sort, binary_search
from general.containers import Counter
from general.stdlib_examples import hashlib

# Use sorting algorithm
sorted_list = quick_sort([64, 34, 25, 12, 22, 11, 90])

# Use binary search
index = binary_search(sorted_list, 25)

# Use container
word_count = Counter("hello world".split())
```

## Algorithm Complexity

| Algorithm     | Time Complexity             | Space Complexity |
| ------------- | --------------------------- | ---------------- |
| Bubble Sort   | O(n²)                       | O(1)             |
| Quick Sort    | O(n log n) avg, O(n²) worst | O(log n)         |
| Merge Sort    | O(n log n)                  | O(n)             |
| Heap Sort     | O(n log n)                  | O(1)             |
| Linear Search | O(n)                        | O(1)             |
| Binary Search | O(log n)                    | O(1)             |

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Examples

### Container Operations

```python
from general.containers import demonstrate_all_containers

# Shows comprehensive examples of:
# - List operations and comprehensions
# - Dictionary manipulation
# - Set theory operations
# - Tuple and named tuple usage
# - Advanced collections (defaultdict, Counter, deque)
demonstrate_all_containers()
```

### Algorithm Demonstrations

```python
from general.algorithms import benchmark_sorting_algorithm, generate_random_array

# Generate test data
test_data = generate_random_array(1000, 1, 1000)

# Benchmark different sorting algorithms
from general.algorithms import quick_sort, merge_sort, heap_sort

result1 = benchmark_sorting_algorithm(quick_sort, test_data)
result2 = benchmark_sorting_algorithm(merge_sort, test_data)
result3 = benchmark_sorting_algorithm(heap_sort, test_data)
```

### Standard Library Usage

```python
from general.stdlib_examples import *

# Demonstrates comprehensive usage of:
# - File system operations
# - Date/time handling
# - Regular expressions
# - JSON processing
# - Cryptographic functions
# - And much more...
demonstrate_all_stdlib()
```

## Educational Value

This module serves as:

- **Learning Resource**: Comprehensive examples of Python features
- **Reference Implementation**: Well-documented algorithm implementations
- **Best Practices Guide**: Demonstrates idiomatic Python code
- **Performance Comparison**: Benchmarking tools for algorithms
- **Standard Library Tour**: Practical examples of built-in modules

Perfect for Python learners, interview preparation, and as a reference for common programming tasks.
