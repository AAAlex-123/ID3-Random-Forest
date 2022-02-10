import time


def timed(prompt: str):
    def decorator(func):
        def inner(*args, **kwargs):
            print(f"Starting procedure: {prompt}")
            start = time.perf_counter()
            rv = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"Process `{prompt}` finished after {end - start} seconds")
            return rv
        return inner
    return decorator


def timed2():
    def decorator(func):
        def inner(*args, **kwargs):
            start = time.perf_counter()
            rv = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"Finished after {end - start} seconds")
            return rv
        return inner
    return decorator
