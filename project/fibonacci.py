def fibonacci(length=7):
    """Generate Fibonacci sequence of given length"""
    if length <= 0:
        return []
    if length == 1:
        return [1]
    fib = [1, 1]
    while len(fib) < length:
        fib.append(fib[-1] + fib[-2])
    return fib[:length]
