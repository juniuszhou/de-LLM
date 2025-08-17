#!/usr/bin/env python3
"""
Demo script for the general Python module.

This script demonstrates the functionality of the general module
including containers, algorithms, and standard library examples.
"""

import sys
import os

# Add the current directory to Python path to import the general module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from general import containers, algorithms, stdlib_examples


def main():
    """Main demo function."""
    print("=" * 60)
    print("    GENERAL PYTHON MODULE DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Demonstrate containers
    print("🔹 CONTAINER OPERATIONS DEMO")
    print("-" * 40)
    try:
        containers.demonstrate_all_containers()
    except Exception as e:
        print(f"Error in containers demo: {e}")
    
    print("\n" + "=" * 60)
    
    # Demonstrate algorithms
    print("🔹 ALGORITHMS DEMO")
    print("-" * 40)
    try:
        algorithms.demonstrate_all_algorithms()
    except Exception as e:
        print(f"Error in algorithms demo: {e}")
    
    print("\n" + "=" * 60)
    
    # Demonstrate standard library
    print("🔹 STANDARD LIBRARY DEMO")
    print("-" * 40)
    try:
        stdlib_examples.demonstrate_all_stdlib()
    except Exception as e:
        print(f"Error in stdlib demo: {e}")
    
    print("\n" + "=" * 60)
    print("    DEMO COMPLETED")
    print("=" * 60)


def interactive_demo():
    """Interactive demo allowing user to choose specific demonstrations."""
    while True:
        print("\n" + "=" * 50)
        print("GENERAL MODULE - INTERACTIVE DEMO")
        print("=" * 50)
        print("1. Container Operations")
        print("2. Sorting Algorithms")
        print("3. Searching Algorithms")
        print("4. Dynamic Programming")
        print("5. Graph Algorithms")
        print("6. Standard Library Examples")
        print("7. Run All Demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            containers.demonstrate_all_containers()
        elif choice == "2":
            algorithms.demonstrate_sorting_algorithms()
        elif choice == "3":
            algorithms.demonstrate_searching_algorithms()
        elif choice == "4":
            algorithms.demonstrate_dynamic_programming()
        elif choice == "5":
            algorithms.demonstrate_graph_algorithms()
        elif choice == "6":
            stdlib_examples.demonstrate_all_stdlib()
        elif choice == "7":
            main()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo script for the general Python module")
    parser.add_argument(
        "-i", "--interactive", 
        action="store_true", 
        help="Run interactive demo"
    )
    parser.add_argument(
        "-m", "--module",
        choices=["containers", "algorithms", "stdlib"],
        help="Run specific module demo"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    elif args.module:
        if args.module == "containers":
            containers.demonstrate_all_containers()
        elif args.module == "algorithms":
            algorithms.demonstrate_all_algorithms()
        elif args.module == "stdlib":
            stdlib_examples.demonstrate_all_stdlib()
    else:
        main()
