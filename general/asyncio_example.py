from asyncio import run
import asyncio


async def say_hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")


async def main():
    print("开始运行 main")
    await asyncio.gather(say_hello(), say_hello(), say_hello())
    print("main 结束")

    # if __name__ == "__main__":
    # Use asyncio.run() to run async main function in regular Python scripts


# print(asyncio.__file__)

asyncio.run(main())
