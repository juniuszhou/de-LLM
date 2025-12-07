"""
Container Usage Examples

This module demonstrates the usage of Python's built-in container types
including lists, dictionaries, sets, tuples, and more advanced containers
from the collections module.
"""

from collections import defaultdict, Counter, deque, namedtuple, OrderedDict
from typing import List, Dict, Set, Tuple, Any, MutableMapping, TypedDict, Union

def typed_dict_example() -> None:
    """Demonstrate typed dictionary operations."""
    print("=== Typed Dictionary Example ===")
    
    class User(TypedDict):
        name: str
        age: int
        email: str
        
    user: User = {
        "name": "John Doe",
        "age": 30,
        "email": "john.doe@example.com"
    }
    print(f"User: {user}")
    
    
def list_operations() -> None:
    """Demonstrate various list operations."""
    print("=== List Operations ===")
    
    # Basic list operations
    numbers = [1, 2, 3, 4, 5]
    print(f"Original list: {numbers}")
    
    # Append and extend
    numbers.append(6)
    numbers.extend([7, 8, 9])
    print(f"After append and extend: {numbers}")
    
    # List comprehensions
    squares = [x**2 for x in numbers if x % 2 == 0]
    print(f"Even squares: {squares}")
    
    # Slicing
    print(f"First 3 elements: {numbers[:3]}")
    print(f"Last 3 elements: {numbers[-3:]}")
    print(f"Every 2nd element: {numbers[::2]}")
    
    # List methods
    numbers.insert(0, 0)
    print(f"After inserting 0 at beginning: {numbers}")
    
    numbers.remove(5)
    print(f"After removing 5: {numbers}")
    
    popped = numbers.pop()
    print(f"Popped element: {popped}, remaining: {numbers}")


def dictionary_operations() -> None:
    """Demonstrate various dictionary operations."""
    print("\n=== Dictionary Operations ===")
    
    # Basic dictionary operations
    student = {
        'name': 'Alice',
        'age': 20,
        'grades': [85, 90, 78, 92]
    }
    print(f"Student info: {student}")
    
    # Dictionary methods
    print(f"Keys: {list(student.keys())}")
    print(f"Values: {list(student.values())}")
    print(f"Items: {list(student.items())}")
    
    # Dictionary comprehensions
    grade_map = {f"subject_{i}": grade for i, grade in enumerate(student['grades'])}
    print(f"Grade mapping: {grade_map}")
    
    # Get with default
    print(f"GPA: {student.get('gpa', 'Not calculated')}")
    
    # Update and merge
    additional_info = {'gpa': 3.6, 'major': 'Computer Science'}
    student.update(additional_info)
    print(f"Updated student: {student}")


def set_operations() -> None:
    """Demonstrate various set operations."""
    print("\n=== Set Operations ===")
    
    # Basic set operations
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}
    
    print(f"Set 1: {set1}")
    print(f"Set 2: {set2}")
    
    # Set operations
    print(f"Union: {set1 | set2}")
    print(f"Intersection: {set1 & set2}")
    print(f"Difference: {set1 - set2}")
    print(f"Symmetric difference: {set1 ^ set2}")
    
    # Set comprehensions
    even_squares = {x**2 for x in range(10) if x % 2 == 0}
    print(f"Even squares: {even_squares}")
    
    # Set methods
    fruits = {'apple', 'banana', 'orange'}
    fruits.add('grape')
    fruits.discard('banana')
    print(f"Fruits: {fruits}")


def tuple_operations() -> None:
    """Demonstrate tuple operations and named tuples."""
    print("\n=== Tuple Operations ===")
    
    # Basic tuple operations
    coordinates = (10, 20)
    print(f"Coordinates: {coordinates}")
    print(f"X: {coordinates[0]}, Y: {coordinates[1]}")
    
    # Tuple unpacking
    x, y = coordinates
    print(f"Unpacked - X: {x}, Y: {y}")
    
    # Named tuples
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(10, 20)
    p2 = Point(x=30, y=40)
    
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Distance from origin for p1: {(p1.x**2 + p1.y**2)**0.5:.2f}")


def advanced_containers() -> None:
    """Demonstrate advanced container types from collections module."""
    print("\n=== Advanced Containers ===")
    
    # defaultdict
    print("defaultdict example:")
    word_count = defaultdict(int)
    text = "hello world hello python world"
    for word in text.split():
        word_count[word] += 1
    print(f"Word count: {dict(word_count)}")
    
    # Counter
    print("\nCounter example:")
    letters = Counter("hello world")
    print(f"Letter frequency: {letters}")
    print(f"Most common 3: {letters.most_common(3)}")
    
    # deque (double-ended queue)
    print("\ndeque example:")
    dq = deque([1, 2, 3])
    dq.appendleft(0)
    dq.append(4)
    print(f"Deque: {dq}")
    
    left = dq.popleft()
    right = dq.pop()
    print(f"After popping left ({left}) and right ({right}): {dq}")
    
    # OrderedDict (though dict is ordered in Python 3.7+)
    print("\nOrderedDict example:")
    od = OrderedDict([('first', 1), ('second', 2), ('third', 3)])
    od.move_to_end('first')
    print(f"Ordered dict after moving 'first' to end: {od}")


def container_performance_tips() -> None:
    """Demonstrate performance considerations for different containers."""
    print("\n=== Container Performance Tips ===")
    
    import time
    
    # List vs Set lookup performance
    large_list = list(range(10000))
    large_set = set(range(10000))
    target = 9999
    
    # List lookup
    start = time.time()
    result = target in large_list
    list_time = time.time() - start
    
    # Set lookup
    start = time.time()
    result = target in large_set
    set_time = time.time() - start
    
    print(f"List lookup time: {list_time:.6f}s")
    print(f"Set lookup time: {set_time:.6f}s")
    print(f"Set is ~{list_time/set_time:.0f}x faster for membership testing")
    
    # Dictionary vs list for key-value storage
    print("\nUse dict for key-value pairs, list for ordered sequences")
    print("Use set for unique elements and fast membership testing")
    print("Use tuple for immutable sequences")
    print("Use deque for frequent additions/removals at both ends")


def demonstrate_all_containers() -> None:
    """Run all container demonstrations."""
    print("Python Container Usage Examples")
    print("=" * 50)
    
    list_operations()
    dictionary_operations()
    set_operations()
    tuple_operations()
    advanced_containers()
    container_performance_tips()


if __name__ == "__main__":
    demonstrate_all_containers()
