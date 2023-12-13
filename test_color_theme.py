# Importing modules
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Numeric types
integer_var = 123
float_var = 123.423
complex_var = 1 + 2j
test = 1 | 2 * complex_var & 5 % 3 @ 325 ^ 364

# String type
string_var = "Hello, Python!"

# List and tuple types
list_var = [1, 2, 3]
tuple_var = (4, 5, 6)

# Set and dictionary types
set_var = {7, 8, 9}
dict_var = {'key1': 'value1', 'key2': 'value2'}

# Boolean type
bool_var_true = True
bool_var_false = False

# NoneType
none_var = None

# Function with various parameters


def test_function(keyword: list, *args, value=None, **kwargs):
    """
    This is a test function.

    Parameters:
        keyword: Some keyword parameter.
        args: Variable positional arguments.
        value: Defaulted keyword argument.
        kwargs: Variable keyword arguments.
    """
    print("Keyword:", keyword)
    print("Variable Positional Arguments:", args)
    print("Defaulted Keyword Argument:", value)
    print("Variable Keyword Arguments:", kwargs)

# Classes


class MyClass:
    def __init__(self, name: str, test: int):
        self.name = name

    def display_name(self):
        print("Name:", self.name)

# Inherited class


class MyDerivedClass(MyClass):
    def __init__(self, name, additional_info):
        super().__init__(name)
        self.additional_info = additional_info

    def display_info(self):
        print("Additional Info:", self.additional_info)


for i in range(100):
    with open('test.txt', 'r') as f:
        f.readlines()

while a < b:
    if a == b:
        return None

# Creating instances
obj1 = MyClass("Object 1")
obj2 = MyDerivedClass("Object 2", "Some additional info")

# Using class methods
obj1.display_name()
obj2.display_name()
obj2.display_info()

# Plotting with matplotlib
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sin Wave')
plt.show()

# Using glob to find files
file_list = glob('*.txt')
print("List of text files:", file_list)
