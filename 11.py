import time
def func_a():
    for _ in range(10):
        time.sleep(1)
        pass

def func_b():
    for _ in range(5):
        func_a()

if __name__ == "__main__":
    func_b()