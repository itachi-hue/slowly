def function(x):
    # Example function, replace with the actual function
    return x**2 - 4*x + 4


def derivative(x):
    # Derivative of the example function, replace with the actual derivative
    return 2*x - 4


if __name__ == '__main__':
    x = 2
    result = derivative(x)
    if result > 0:
        print(f'The function is increasing at x = {x}.')
    elif result < 0:
        print(f'The function is decreasing at x = {x}.')
    else:
        print(f'The function is neither increasing nor decreasing at x = {x}.')
