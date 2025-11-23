#!/usr/bin/env python3
"""
Example: How to Set Log Level Before Program Starts

This demonstrates the most common and best practices for setting
log levels before your program begins execution.
"""

import logging
import os
import sys


# ============================================================================
# METHOD 1: Using basicConfig() - Most Common
# ============================================================================
def method1_basic_config():
    """Set log level using basicConfig() at program start."""
    print("\n" + "=" * 70)
    print("METHOD 1: Using basicConfig()")
    print("=" * 70)
    
    # IMPORTANT: Call this BEFORE any logging.getLogger() calls
    logging.basicConfig(
        level=logging.INFO,  # Set level here
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.debug("DEBUG message (won't show - level is INFO)")
    logger.info("INFO message (will show)")
    logger.warning("WARNING message (will show)")


# ============================================================================
# METHOD 2: From Environment Variable - Best for Production
# ============================================================================
def method2_environment_variable():
    """Set log level from environment variable."""
    print("\n" + "=" * 70)
    print("METHOD 2: From Environment Variable")
    print("=" * 70)
    
    # Get log level from environment, default to WARNING
    log_level = os.environ.get('LOG_LEVEL', 'WARNING').upper()
    
    # Convert string to logging level constant
    numeric_level = getattr(logging, log_level, logging.WARNING)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(levelname)-8s: %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.debug("DEBUG message")
    logger.info("INFO message")
    logger.warning("WARNING message")
    
    print(f"\nCurrent LOG_LEVEL environment variable: {os.environ.get('LOG_LEVEL', 'Not set')}")
    print(f"Using log level: {log_level} ({numeric_level})")
    print("\nTo test: LOG_LEVEL=DEBUG python logging_setup_example.py")


# ============================================================================
# METHOD 3: From Command-Line Argument - Best for Flexibility
# ============================================================================
def method3_command_line():
    """Set log level from command-line argument."""
    print("\n" + "=" * 70)
    print("METHOD 3: From Command-Line Argument")
    print("=" * 70)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Example with log level argument')
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(levelname)-8s: %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.debug("DEBUG message")
    logger.info("INFO message")
    logger.warning("WARNING message")
    
    print(f"\nUsing log level from command-line: {args.log_level}")
    print("To test: python logging_setup_example.py --log-level DEBUG")


# ============================================================================
# METHOD 4: Priority-Based Setup (Command-line > Env > Default)
# ============================================================================
def method4_priority_based():
    """Set log level with priority: command-line > environment > default."""
    print("\n" + "=" * 70)
    print("METHOD 4: Priority-Based Setup")
    print("=" * 70)
    
    # Default log level
    log_level = 'INFO'
    
    # Check command-line arguments first (highest priority)
    if '--log-level' in sys.argv:
        idx = sys.argv.index('--log-level')
        if idx + 1 < len(sys.argv):
            log_level = sys.argv[idx + 1]
    # Check environment variable second
    elif 'LOG_LEVEL' in os.environ:
        log_level = os.environ['LOG_LEVEL']
    # Otherwise use default
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.debug("DEBUG message")
    logger.info("INFO message")
    logger.warning("WARNING message")
    
    print(f"\nLog level determined: {log_level.upper()}")
    print("Priority: Command-line > Environment > Default")


# ============================================================================
# METHOD 5: Set Root Logger Level Directly
# ============================================================================
def method5_root_logger():
    """Set root logger level directly."""
    print("\n" + "=" * 70)
    print("METHOD 5: Set Root Logger Level Directly")
    print("=" * 70)
    
    # Set root logger level
    logging.root.setLevel(logging.DEBUG)
    
    # Configure basic handler if not already configured
    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)-8s: %(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
    
    logger = logging.getLogger(__name__)
    logger.debug("DEBUG message (will show)")
    logger.info("INFO message (will show)")
    logger.warning("WARNING message (will show)")


# ============================================================================
# COMPLETE EXAMPLE: Production-Ready Setup Function
# ============================================================================
def setup_logging(log_level=None, format_string=None):
    """
    Production-ready logging setup function.
    
    Args:
        log_level: Log level string (DEBUG, INFO, etc.) or None to auto-detect
        format_string: Custom format string or None for default
    """
    # Determine log level with priority: argument > env > default
    if log_level is None:
        if '--log-level' in sys.argv:
            idx = sys.argv.index('--log-level')
            log_level = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'INFO'
        elif 'LOG_LEVEL' in os.environ:
            log_level = os.environ['LOG_LEVEL']
        else:
            log_level = 'INFO'
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any existing configuration
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# MAIN: Demonstrate All Methods
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("HOW TO SET LOG LEVEL BEFORE PROGRAM STARTS")
    print("=" * 70)
    
    # Reset logging to show each method independently
    logging.root.handlers = []
    logging.root.setLevel(logging.WARNING)
    
    # Method 1: Basic Config
    method1_basic_config()
    
    # Method 2: Environment Variable
    logging.root.handlers = []
    method2_environment_variable()
    
    # Method 3: Command-line (commented out to avoid argparse conflicts)
    # logging.root.handlers = []
    # method3_command_line()
    
    # Method 4: Priority-based
    logging.root.handlers = []
    method4_priority_based()
    
    # Method 5: Root logger
    logging.root.handlers = []
    method5_root_logger()
    
    print("\n" + "=" * 70)
    print("PRODUCTION-READY EXAMPLE")
    print("=" * 70)
    logging.root.handlers = []
    logger = setup_logging()
    logger.info("Program started with production-ready logging setup")
    logger.debug("This debug message won't show (level is INFO)")
    
    print("\n" + "=" * 70)
    print("QUICK REFERENCE")
    print("=" * 70)
    print("""
1. Set log level at the VERY START of your program
2. Use basicConfig() - it only works the first time it's called
3. Priority order: Command-line > Environment > Default
4. Common usage:
   - Development: logging.basicConfig(level=logging.DEBUG)
   - Production: LOG_LEVEL=INFO python script.py
   - Testing: python script.py --log-level DEBUG
    """)

