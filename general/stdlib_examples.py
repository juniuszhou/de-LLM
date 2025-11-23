"""
Python Standard Library Examples

This module demonstrates the usage of various Python standard library modules
including os, sys, datetime, json, re, itertools, functools, and more.
"""

import os
import sys
import datetime
import json
import re
import itertools
import functools
import math
import random
import string
import pathlib
import urllib.parse
import base64
import hashlib
import uuid
import logging
from typing import List, Dict, Any, Iterator, Callable


def os_module_examples() -> None:
    """Demonstrate os module functionality."""
    print("=== OS Module Examples ===")
    
    # Current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Environment variables
    print(f"PATH environment variable: {os.environ.get('PATH', 'Not found')[:100]}...")
    print(f"HOME directory: {os.environ.get('HOME', 'Not found')}")
    
    # File and directory operations
    print(f"Directory contents: {os.listdir('.')[:5]}...")  # First 5 items
    
    # Path operations
    sample_path = "/home/user/documents/file.txt"
    print(f"Sample path: {sample_path}")
    print(f"Directory: {os.path.dirname(sample_path)}")
    print(f"Filename: {os.path.basename(sample_path)}")
    print(f"File extension: {os.path.splitext(sample_path)[1]}")
    print(f"Path exists: {os.path.exists(sample_path)}")


def sys_module_examples() -> None:
    """Demonstrate sys module functionality."""
    print("\n=== SYS Module Examples ===")
    
    # System information
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Python executable: {sys.executable}")
    
    # Command line arguments (would be empty in this context)
    print(f"Command line arguments: {sys.argv}")
    
    # Python path
    print(f"Python path (first 3): {sys.path[:3]}")
    
    # Memory usage
    import sys
    sample_list = list(range(1000))
    print(f"Size of list with 1000 integers: {sys.getsizeof(sample_list)} bytes")


def datetime_module_examples() -> None:
    """Demonstrate datetime module functionality."""
    print("\n=== DateTime Module Examples ===")
    
    # Current date and time
    now = datetime.datetime.now()
    today = datetime.date.today()
    current_time = datetime.time(now.hour, now.minute, now.second)
    
    print(f"Current datetime: {now}")
    print(f"Today's date: {today}")
    print(f"Current time: {current_time}")
    
    # Formatting dates
    print(f"Formatted date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Formatted date (readable): {now.strftime('%B %d, %Y at %I:%M %p')}")
    
    # Date arithmetic
    tomorrow = today + datetime.timedelta(days=1)
    last_week = today - datetime.timedelta(weeks=1)
    print(f"Tomorrow: {tomorrow}")
    print(f"Last week: {last_week}")
    
    # Parsing dates
    date_string = "2024-01-15 14:30:00"
    parsed_date = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    print(f"Parsed date: {parsed_date}")


def json_module_examples() -> None:
    """Demonstrate json module functionality."""
    print("\n=== JSON Module Examples ===")
    
    # Python object to JSON
    data = {
        "name": "Alice",
        "age": 30,
        "city": "New York",
        "skills": ["Python", "JavaScript", "SQL"],
        "is_student": False,
        "graduation_date": None
    }
    
    json_string = json.dumps(data, indent=2)
    print("Python to JSON:")
    print(json_string)
    
    # JSON to Python object
    parsed_data = json.loads(json_string)
    print(f"\nJSON to Python: {parsed_data}")
    print(f"Name: {parsed_data['name']}")
    print(f"Skills: {', '.join(parsed_data['skills'])}")


def regex_examples() -> None:
    """Demonstrate regular expressions."""
    print("\n=== Regular Expression Examples ===")
    
    text = "Contact us at support@company.com or sales@company.com. Phone: +1-555-123-4567"
    
    # Find email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    print(f"Email addresses found: {emails}")
    
    # Find phone numbers
    phone_pattern = r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    print(f"Phone numbers found: {phones}")
    
    # Substitute text
    censored_text = re.sub(email_pattern, '[EMAIL]', text)
    print(f"Censored text: {censored_text}")
    
    # Split text
    words = re.split(r'\W+', text)
    print(f"Words: {words[:10]}...")  # First 10 words


def itertools_examples() -> None:
    """Demonstrate itertools module functionality."""
    print("\n=== Itertools Examples ===")
    
    # Infinite iterators
    print("Count iterator (first 5):", list(itertools.islice(itertools.count(10, 2), 5)))
    print("Cycle iterator (first 8):", list(itertools.islice(itertools.cycle(['A', 'B', 'C']), 8)))
    print("Repeat iterator (first 5):", list(itertools.islice(itertools.repeat('Hello', 5), 5)))
    
    # Combinatorial iterators
    items = ['A', 'B', 'C']
    print(f"Permutations of {items}: {list(itertools.permutations(items, 2))}")
    print(f"Combinations of {items}: {list(itertools.combinations(items, 2))}")
    print(f"Combinations with replacement: {list(itertools.combinations_with_replacement(items, 2))}")
    print(f"Product: {list(itertools.product(items, [1, 2]))}")
    
    # Functional iterators
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Accumulate (cumulative sum): {list(itertools.accumulate(numbers))}")
    print(f"Filter false (even numbers): {list(itertools.filterfalse(lambda x: x % 2, numbers))}")
    
    # Grouping
    data = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]
    grouped = itertools.groupby(data, key=lambda x: x[0])
    print("Grouped data:")
    for key, group in grouped:
        print(f"  {key}: {list(group)}")


def functools_examples() -> None:
    """Demonstrate functools module functionality."""
    print("\n=== Functools Examples ===")
    
    # Partial functions
    def multiply(x: int, y: int) -> int:
        return x * y
    
    double = functools.partial(multiply, 2)
    triple = functools.partial(multiply, 3)
    
    print(f"Double 5: {double(5)}")
    print(f"Triple 5: {triple(5)}")
    
    # Reduce
    numbers = [1, 2, 3, 4, 5]
    product = functools.reduce(lambda x, y: x * y, numbers)
    print(f"Product of {numbers}: {product}")
    
    # LRU Cache decorator
    @functools.lru_cache(maxsize=128)
    def expensive_function(n: int) -> int:
        """Simulate an expensive computation."""
        print(f"Computing for {n}...")
        return n * n * n
    
    print("First call:")
    result1 = expensive_function(5)
    print(f"Result: {result1}")
    
    print("Second call (cached):")
    result2 = expensive_function(5)
    print(f"Result: {result2}")
    
    print(f"Cache info: {expensive_function.cache_info()}")


def math_module_examples() -> None:
    """Demonstrate math module functionality."""
    print("\n=== Math Module Examples ===")
    
    # Basic math functions
    print(f"Square root of 16: {math.sqrt(16)}")
    print(f"2 to the power of 3: {math.pow(2, 3)}")
    print(f"Ceiling of 4.3: {math.ceil(4.3)}")
    print(f"Floor of 4.7: {math.floor(4.7)}")
    print(f"Factorial of 5: {math.factorial(5)}")
    
    # Trigonometric functions
    angle = math.pi / 4  # 45 degrees in radians
    print(f"Sin(45°): {math.sin(angle):.4f}")
    print(f"Cos(45°): {math.cos(angle):.4f}")
    print(f"Tan(45°): {math.tan(angle):.4f}")
    
    # Logarithmic functions
    print(f"Natural log of e: {math.log(math.e)}")
    print(f"Log base 10 of 100: {math.log10(100)}")
    print(f"Log base 2 of 8: {math.log2(8)}")
    
    # Constants
    print(f"Pi: {math.pi}")
    print(f"Euler's number: {math.e}")
    print(f"Infinity: {math.inf}")


def pathlib_examples() -> None:
    """Demonstrate pathlib module functionality."""
    print("\n=== Pathlib Examples ===")
    
    # Create a Path object
    current_path = pathlib.Path.cwd()
    print(f"Current directory: {current_path}")
    
    # Path operations
    sample_path = pathlib.Path("/home/user/documents/file.txt")
    print(f"Sample path: {sample_path}")
    print(f"Parent directory: {sample_path.parent}")
    print(f"Filename: {sample_path.name}")
    print(f"Stem (filename without extension): {sample_path.stem}")
    print(f"Suffix (extension): {sample_path.suffix}")
    
    # Path joining
    new_path = current_path / "subdirectory" / "file.txt"
    print(f"Joined path: {new_path}")
    
    # Path properties
    print(f"Is absolute: {sample_path.is_absolute()}")
    print(f"Exists: {sample_path.exists()}")


def string_module_examples() -> None:
    """Demonstrate string module functionality."""
    print("\n=== String Module Examples ===")
    
    # String constants
    print(f"ASCII letters: {string.ascii_letters}")
    print(f"Digits: {string.digits}")
    print(f"Punctuation: {string.punctuation}")
    print(f"Printable characters: {len(string.printable)} total")
    
    # Generate random strings
    random_password = ''.join(random.choices(
        string.ascii_letters + string.digits + string.punctuation, 
        k=12
    ))
    print(f"Random password: {random_password}")
    
    # String formatting
    template = string.Template("Hello $name, welcome to $place!")
    formatted = template.substitute(name="Alice", place="Python")
    print(f"Template formatting: {formatted}")


def crypto_examples() -> None:
    """Demonstrate cryptographic and encoding functions."""
    print("\n=== Cryptographic Examples ===")
    
    # Base64 encoding/decoding
    text = "Hello, World!"
    encoded = base64.b64encode(text.encode()).decode()
    decoded = base64.b64decode(encoded.encode()).decode()
    print(f"Original: {text}")
    print(f"Base64 encoded: {encoded}")
    print(f"Base64 decoded: {decoded}")
    
    # Hashing
    sha256_hash = hashlib.sha256(text.encode()).hexdigest()
    md5_hash = hashlib.md5(text.encode()).hexdigest()
    print(f"SHA256 hash: {sha256_hash}")
    print(f"MD5 hash: {md5_hash}")
    
    # UUID generation
    random_uuid = uuid.uuid4()
    name_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, 'python.org')
    print(f"Random UUID: {random_uuid}")
    print(f"Name-based UUID: {name_uuid}")


def url_examples() -> None:
    """Demonstrate URL parsing and manipulation."""
    print("\n=== URL Examples ===")
    
    # Parse URL
    url = "https://www.example.com:8080/path/to/page?param1=value1&param2=value2#section"
    parsed = urllib.parse.urlparse(url)
    
    print(f"Original URL: {url}")
    print(f"Scheme: {parsed.scheme}")
    print(f"Hostname: {parsed.hostname}")
    print(f"Port: {parsed.port}")
    print(f"Path: {parsed.path}")
    print(f"Query: {parsed.query}")
    print(f"Fragment: {parsed.fragment}")
    
    # Parse query parameters
    query_params = urllib.parse.parse_qs(parsed.query)
    print(f"Query parameters: {query_params}")
    
    # URL encoding/decoding
    text_to_encode = "Hello World! & Special Characters"
    encoded_url = urllib.parse.quote(text_to_encode)
    decoded_url = urllib.parse.unquote(encoded_url)
    print(f"URL encoded: {encoded_url}")
    print(f"URL decoded: {decoded_url}")


def random_module_examples() -> None:
    """Demonstrate random module functionality."""
    print("\n=== Random Module Examples ===")
    
    # Set seed for reproducible results
    random.seed(42)
    
    # Random numbers
    print(f"Random float [0.0, 1.0): {random.random():.4f}")
    print(f"Random integer [1, 10]: {random.randint(1, 10)}")
    print(f"Random float [1.0, 10.0]: {random.uniform(1.0, 10.0):.4f}")
    
    # Random choices
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    print(f"Random choice: {random.choice(colors)}")
    print(f"Random sample (3): {random.sample(colors, 3)}")
    print(f"Random choices with replacement (5): {random.choices(colors, k=5)}")
    
    # Shuffle
    numbers = list(range(10))
    random.shuffle(numbers)
    print(f"Shuffled numbers: {numbers}")
    
    # Random distributions
    print(f"Gaussian (mean=0, std=1): {random.gauss(0, 1):.4f}")
    print(f"Exponential (lambda=1): {random.expovariate(1):.4f}")


def logging_module_examples() -> None:
    """Demonstrate logging module functionality."""
    print("\n=== Logging Module Examples ===")
    
    print("\n--- Method 1: Using basicConfig() at program start ---")
    print("""
# Set log level BEFORE any logging calls
import logging

# Option A: Set level directly
logging.basicConfig(level=logging.DEBUG)

# Option B: Set level from string (useful for config files)
logging.basicConfig(level=logging.getLevelName('INFO'))

# Option C: Set level from environment variable
import os
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level.upper()))
    """)
    
    # Demonstrate different log levels
    print("\n--- Log Levels (from lowest to highest) ---")
    levels = [
        ('DEBUG', logging.DEBUG, 'Detailed diagnostic info'),
        ('INFO', logging.INFO, 'General informational messages'),
        ('WARNING', logging.WARNING, 'Warning messages'),
        ('ERROR', logging.ERROR, 'Error messages'),
        ('CRITICAL', logging.CRITICAL, 'Critical error messages'),
    ]
    
    for level_name, level_value, description in levels:
        print(f"  {level_name:10s} ({level_value:2d}): {description}")
    
    print("\n--- Setting Log Level Before Program Start ---")
    print("""
# Method 1: Using basicConfig() - MUST be called before any logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the root logger level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Method 2: Set root logger level directly
logging.root.setLevel(logging.INFO)

# Method 3: From environment variable (common in production)
import os
log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level))

# Method 4: From command-line argument
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log-level', default='INFO', 
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
args = parser.parse_args()
logging.basicConfig(level=getattr(logging, args.log_level))

# Method 5: From configuration file
import json
with open('config.json') as f:
    config = json.load(f)
logging.basicConfig(level=getattr(logging, config['log_level']))
    """)
    
    # Demonstrate actual logging with different levels
    print("\n--- Example: Logging with Different Levels ---")
    
    # Set up a temporary logger for demonstration
    demo_logger = logging.getLogger('demo')
    demo_logger.setLevel(logging.DEBUG)
    
    # Create a console handler if basicConfig hasn't been called
    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)-8s: %(message)s')
        handler.setFormatter(formatter)
        demo_logger.addHandler(handler)
        demo_logger.propagate = False
    
    print("Demonstrating log messages (if level allows):")
    demo_logger.debug("This is a DEBUG message")
    demo_logger.info("This is an INFO message")
    demo_logger.warning("This is a WARNING message")
    demo_logger.error("This is an ERROR message")
    demo_logger.critical("This is a CRITICAL message")
    
    print("\n--- Best Practices ---")
    print("""
1. Set log level at the VERY START of your program, before any imports that use logging
2. Use environment variables for production: LOG_LEVEL=INFO python script.py
3. Use command-line arguments for flexibility: python script.py --log-level DEBUG
4. Use basicConfig() only once (it only works the first time)
5. For libraries, use getLogger(__name__) instead of root logger
    """)
    
    # Show practical example
    print("\n--- Practical Example: Setting Log Level Before Program Start ---")
    print("""
# File: my_program.py
import logging
import os
import sys

# Set log level BEFORE anything else
def setup_logging():
    # Priority: command-line arg > environment variable > default
    log_level = 'INFO'  # default
    
    if '--log-level' in sys.argv:
        idx = sys.argv.index('--log-level')
        log_level = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'INFO'
    elif 'LOG_LEVEL' in os.environ:
        log_level = os.environ['LOG_LEVEL']
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Call setup_logging() at the very beginning
if __name__ == '__main__':
    setup_logging()  # Set log level FIRST
    
    logger = logging.getLogger(__name__)
    logger.info("Program started")
    logger.debug("This won't show if level is INFO or higher")
    
    # Rest of your program...
    """)


def demonstrate_all_stdlib() -> None:
    """Run all standard library demonstrations."""
    print("Python Standard Library Examples")
    print("=" * 50)
    
    os_module_examples()
    sys_module_examples()
    datetime_module_examples()
    json_module_examples()
    regex_examples()
    itertools_examples()
    functools_examples()
    math_module_examples()
    pathlib_examples()
    string_module_examples()
    crypto_examples()
    url_examples()
    random_module_examples()
    logging_module_examples()


if __name__ == "__main__":
    demonstrate_all_stdlib()
