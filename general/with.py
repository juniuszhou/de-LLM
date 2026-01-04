from contextlib import contextmanager


@contextmanager
def context_manager():
    print("Entering context")
    # execute the code via yield
    yield
    print("Exiting context")


with context_manager() as cm:
    print("Inside with block")


# define a class-based context manager
class context_manager_class:
    """Class-based context manager - no decorator needed."""

    def __init__(self):
        self.func = None

    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context")
        return False  # Don't suppress exceptions

    def __call__(self, func):
        print("Calling context manager")
        return func


with context_manager_class() as cm:
    print("Inside with block")

print("Outside with block")
