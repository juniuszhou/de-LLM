import numpy as np


def f(x):  # 目标函数
    return x**2


def f_grad(x):  # 目标函数的梯度(导数)
    return 2 * x


def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f"epoch 10, x: {x:f}")
    return results


results = gd(0.2, f_grad)
print(results)
