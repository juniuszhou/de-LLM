import asyncio
import threading
import time

async def async_function():
    print("Async function started")
    await asyncio.sleep(1)
    print("Async function finished")

async def main():
    print("Hello, world!")
    await async_function()
    
def main():
    print("===== Main thread: Starting async function in a new thread...")

def run_async_in_thread():
    """Run async function in a new thread"""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_function())
    finally:
        loop.close()

if __name__ == "__main__":
    print("Main thread: Starting async function in a new thread...")
    
    # Create and start a new thread
    thread = threading.Thread(target=run_async_in_thread, daemon=False)
    thread.start()
    
    print("Main thread: Thread started, continuing with other work...")
    time.sleep(0.5)  # Simulate other work in main thread
    print("Main thread: Doing other work...")
    
    # Wait for the thread to complete
    thread.join()
    print("Main thread: Thread completed!")

