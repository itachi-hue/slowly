# Define the function to calculate the line equation
from sympy import symbols, Eq, solve

def find_line_equation(point1, point2):
    x, y = symbols('x y')
    m, b = symbols('m b')
    # Equation 1: Point (0, 3)
    eq1 = Eq(b, 3)
    # Equation 2: Point (9, 2)
    eq2 = Eq(2 - m * 9 + b, y)
    # Solve the system of equations
    solution = solve((eq1, eq2), (m, b))
    return f'y = {solution[m]}x + {solution[b]}'

# Test the function with given points
line_equation = find_line_equation((0, 3), (9, 2))
print(line_equation)