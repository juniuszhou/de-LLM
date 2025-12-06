"""
Algorithm Implementations

This module contains implementations of common algorithms including
sorting, searching, and other fundamental computer science algorithms.
"""

from typing import List, Optional, Tuple, Any
import random
import time
import unittest
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result
    return wrapper


# Sorting Algorithms
def bubble_sort(arr: List[int]) -> List[int]:
    """
    Bubble sort implementation.
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    arr = arr.copy()  # Don't modify original
    n = len(arr)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: early exit if array is sorted
            break
    
    return arr


def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick sort implementation.
    Time Complexity: O(n log n) average, O(n²) worst
    Space Complexity: O(log n)
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge sort implementation.
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Helper function for merge sort."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def heap_sort(arr: List[int]) -> List[int]:
    """
    Heap sort implementation.
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    arr = arr.copy()
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Swap
        heapify(arr, i, 0)
    
    return arr


def heapify(arr: List[int], n: int, i: int) -> None:
    """Helper function for heap sort."""

    text: str = ""
    b: bool = True
    c: dict = {}
    d: list = []
    e: tuple = ()
    f: set = set()
    g: int = 0
    h: float = 0.0
    i: complex = 0j
    j: range = range(10)
    k: slice = slice(10)
    l: bytes = b""
    m: bytearray = bytearray(10)
    n: memoryview = memoryview(b"")
    o: None = None
    p: Any = None
    q: Optional[int] = None
    q = 10
    left: int = 2 * i + 1
    right: int = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


# Searching Algorithms
def linear_search(arr: List[int], target: int) -> Optional[int]:
    """
    Linear search implementation.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    for i, value in enumerate(arr):
        if value == target:
            return i
    return None


def binary_search(arr: List[int], target: int) -> Optional[int]:
    """
    Binary search implementation (requires sorted array).
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return None


def binary_search_recursive(arr: List[int], target: int, left: int = 0, right: int = None) -> Optional[int]:
    """
    Recursive binary search implementation.
    Time Complexity: O(log n)
    Space Complexity: O(log n)
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return None
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


# Dynamic Programming Examples
def fibonacci(n: int) -> int:
    """
    Fibonacci using dynamic programming.
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def fibonacci_memoized(n: int, memo: dict = None) -> int:
    """
    Fibonacci with memoization.
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find length of longest common subsequence using dynamic programming.
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


# Graph Algorithms
def depth_first_search(graph: dict, start: str, visited: set = None) -> List[str]:
    """
    Depth-First Search implementation.
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    path = [start]
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            path.extend(depth_first_search(graph, neighbor, visited))
    
    return path


def breadth_first_search(graph: dict, start: str) -> List[str]:
    """
    Breadth-First Search implementation.
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    from collections import deque
    
    visited = set()
    queue = deque([start])
    path = []
    
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            path.append(vertex)
            
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return path


# Utility Functions
def generate_random_array(size: int, min_val: int = 1, max_val: int = 100) -> List[int]:
    """Generate a random array for testing algorithms."""
    return [random.randint(min_val, max_val) for _ in range(size)]


def is_sorted(arr: List[int]) -> bool:
    """Check if an array is sorted."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


@timing_decorator
def benchmark_sorting_algorithm(sort_func, arr: List[int]) -> List[int]:
    """Benchmark a sorting algorithm."""
    return sort_func(arr)


def demonstrate_sorting_algorithms() -> None:
    """Demonstrate all sorting algorithms."""
    print("=== Sorting Algorithms Demo ===")
    
    # Generate test data
    test_array = generate_random_array(20, 1, 100)
    print(f"Original array: {test_array}")
    
    algorithms = [
        ("Bubble Sort", bubble_sort),
        ("Quick Sort", quick_sort),
        ("Merge Sort", merge_sort),
        ("Heap Sort", heap_sort),
        ("Python Built-in", sorted)
    ]
    
    for name, algo in algorithms:
        result = benchmark_sorting_algorithm(algo, test_array)
        print(f"{name}: {result[:10]}{'...' if len(result) > 10 else ''}")
        print(f"Is sorted: {is_sorted(result)}")
        print()


def demonstrate_searching_algorithms() -> None:
    """Demonstrate searching algorithms."""
    print("=== Searching Algorithms Demo ===")
    
    # Generate and sort test data
    test_array = sorted(generate_random_array(20, 1, 100))
    target = test_array[10]  # Pick an element that exists
    
    print(f"Sorted array: {test_array}")
    print(f"Searching for: {target}")
    
    # Linear search
    linear_result = linear_search(test_array, target)
    print(f"Linear search result: index {linear_result}")
    
    # Binary search
    binary_result = binary_search(test_array, target)
    print(f"Binary search result: index {binary_result}")
    
    # Recursive binary search
    recursive_result = binary_search_recursive(test_array, target)
    print(f"Recursive binary search result: index {recursive_result}")


def demonstrate_dynamic_programming() -> None:
    """Demonstrate dynamic programming algorithms."""
    print("\n=== Dynamic Programming Demo ===")
    
    # Fibonacci
    n = 10
    print(f"Fibonacci({n}) = {fibonacci(n)}")
    print(f"Fibonacci({n}) memoized = {fibonacci_memoized(n)}")
    
    # Longest Common Subsequence
    text1, text2 = "ABCDGH", "AEDFHR"
    lcs_length = longest_common_subsequence(text1, text2)
    print(f"LCS of '{text1}' and '{text2}': length = {lcs_length}")


def demonstrate_graph_algorithms() -> None:
    """Demonstrate graph algorithms."""
    print("\n=== Graph Algorithms Demo ===")
    
    # Sample graph representation (adjacency list)
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    
    print(f"Graph: {graph}")
    
    # DFS
    dfs_path = depth_first_search(graph, 'A')
    print(f"DFS traversal from A: {dfs_path}")
    
    # BFS
    bfs_path = breadth_first_search(graph, 'A')
    print(f"BFS traversal from A: {bfs_path}")


def demonstrate_all_algorithms() -> None:
    """Run all algorithm demonstrations."""
    print("Python Algorithm Implementations")
    print("=" * 50)
    
    demonstrate_sorting_algorithms()
    demonstrate_searching_algorithms()
    demonstrate_dynamic_programming()
    demonstrate_graph_algorithms()


class TestBubbleSort(unittest.TestCase):
    """Unit tests for bubble_sort function."""
    
    def test_empty_array(self):
        """Test bubble sort with empty array."""
        result = bubble_sort([])
        self.assertEqual(result, [])
    
    def test_single_element(self):
        """Test bubble sort with single element."""
        result = bubble_sort([42])
        self.assertEqual(result, [42])
    
    def test_already_sorted(self):
        """Test bubble sort with already sorted array."""
        arr = [1, 2, 3, 4, 5]
        result = bubble_sort(arr)
        self.assertEqual(result, [1, 2, 3, 4, 5])
        # Verify original array is not modified
        self.assertEqual(arr, [1, 2, 3, 4, 5])
    
    def test_reverse_sorted(self):
        """Test bubble sort with reverse sorted array."""
        result = bubble_sort([5, 4, 3, 2, 1])
        self.assertEqual(result, [1, 2, 3, 4, 5])
    
    def test_random_array(self):
        """Test bubble sort with random unsorted array."""
        result = bubble_sort([64, 34, 25, 12, 22, 11, 90])
        self.assertEqual(result, [11, 12, 22, 25, 34, 64, 90])
    
    def test_duplicates(self):
        """Test bubble sort with duplicate elements."""
        result = bubble_sort([3, 1, 4, 1, 5, 9, 2, 6, 5])
        self.assertEqual(result, [1, 1, 2, 3, 4, 5, 5, 6, 9])
    
    def test_all_same_elements(self):
        """Test bubble sort with all identical elements."""
        result = bubble_sort([7, 7, 7, 7, 7])
        self.assertEqual(result, [7, 7, 7, 7, 7])
    
    def test_two_elements_sorted(self):
        """Test bubble sort with two elements already sorted."""
        result = bubble_sort([1, 2])
        self.assertEqual(result, [1, 2])
    
    def test_two_elements_unsorted(self):
        """Test bubble sort with two elements unsorted."""
        result = bubble_sort([2, 1])
        self.assertEqual(result, [1, 2])
    
    def test_negative_numbers(self):
        """Test bubble sort with negative numbers."""
        result = bubble_sort([-3, -1, -4, -1, -5, 0, 2])
        self.assertEqual(result, [-5, -4, -3, -1, -1, 0, 2])
    
    def test_original_array_unchanged(self):
        """Test that original array is not modified."""
        original = [3, 1, 4, 1, 5]
        original_copy = original.copy()
        result = bubble_sort(original)
        self.assertEqual(original, original_copy)  # Original unchanged
        self.assertEqual(result, [1, 1, 3, 4, 5])  # Result is sorted


def run_bubble_sort_tests():
    """Run all bubble sort unit tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBubbleSort)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

# python general/algorithms.py test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests if "test" argument is provided
        print("Running Bubble Sort Unit Tests...")
        print("=" * 50)
        result = run_bubble_sort_tests()
        print("=" * 50)
        if result.wasSuccessful():
            print("All tests passed! ✅")
        else:
            print("Some tests failed! ❌")
            sys.exit(1)
