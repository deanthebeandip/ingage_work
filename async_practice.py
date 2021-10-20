import asyncio

loop = asyncio.get_event_loop()

async def while_loop():
    n = 0
    while True:
        print(f"{n}")
        await asyncio.sleep(2)
        n = n+1

async def some_func():
    print("Entered Func")
    await asyncio.sleep(5)
    print("Leaving Func")

future = loop.create_task(while_loop())
loop.run_until_complete(some_func())
