# Datacamp OBJECT ORIENTED PROGRAMMING IN PYTHON

# Chapter 1: Getting ready for object-oriented programming
	
    # Creating Functions
    # Creating a complex data type
    # Create a function that returns a NumPy array
    # Creating a class

# Chapter 2: Deep dive into classes and objects

    # Object: Instance of a Class
    # The Init Method
    # Instance Variables
    # Multiple Instance Variables
    # Class Variables
    # Overriding Class Variables
    # Methods I
    # Methods II
    # Methods III

# Chapter 3: Fancy classes, fancy objects

    # Return Statement I
    # Return Statement II: The Return of the DataShell
    # Return Statement III: A More Powerful DataShell
    # Data as Attribute
    # Renaming Columns
    # Self-Describing DataShells

# Chapter 4: Inheritance, polymorphism and composition

    # Animal Inheritance
    # Vertebrate Inheritance
    # Abstract Class DataShell I
    # Abstract Class DataShell II















######## Chapter 1: Getting ready for object-oriented programming















### Creating Functions

# In this exercise, we will review functions, as they are key building blocks of 
# object-oriented programs. For this, we will create a simple function average_numbers() which averages a list of numbers. Remember 
# that lists are a basic data type in Python that we can build using the [] bracket notation.
# Here is an example of a function that returns the square of an integer:

# def square_function(x):
#     x_squared =  x**2
#     return x_squared

# Create function that returns the average of an integer list
def average_numbers(num_list): 
    avg = sum(num_list)/float(len(num_list)) # divide by length of list
    return avg

# Take the average of a list: my_avg
my_avg = average_numbers([1,2,3,4,5,6])

# Print out my_avg
print(my_avg)






### Creating a complex data type

# In this exercise, we'll take a closer look at the flexibility of the list data 
# type, by creating a list of lists. In Python, lists usually look like our list 
# example below, and can be made up of either simple strings, integers, or a combination of both.
# list = [1,2]
# In creating a list of lists, we're building up to the concept of a NumPy array.

# Create a list that contains two lists: matrix
matrix = [[1,2,3,4], [5,6,7,8]]

# Print the matrix list
print(matrix)







### Create a function that returns a NumPy array

# In this exercise, we'll continue working with the numpy package and our previous structures.
# We'll create a NumPy array of the float (numerical) data type so that we can work with 
# a multi-dimensional data objects, much like columns and rows in a spreadsheet.

# Import numpy as np
import numpy as np

# List input: my_matrix
my_matrix = [[1,2,3,4], [5,6,7,8]] 

# Function that converts lists to arrays: return_array
def return_array(matrix):
    array = np.array(matrix, dtype = float)
    return array
    
# Call return_array on my_matrix, and print the output
print(return_array(my_matrix))

<script.py> output:
    [[1. 2. 3. 4.]
     [5. 6. 7. 8.]]







### Creating a class

# We're going to be working on building a class, which is a way to organize functions 
# and variables in Python. To start with, let's look at the simplest possible way to create a class.

# Create a class: DataShell
class DataShell(): 
    pass















######## Chapter 2: Deep dive into classes and objects















### Object: Instance of a Class

# As we learned earlier, a class is like a blueprint: we can make many copies of our class.
# When we do this, we say that we are instantiating our class. These instances are called objects.
# Here is an example of class instantiation:
# object_name = ClassName()

# Create empty class: DataShell
class DataShell:
  
    # Pass statement
    pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)

output:
<__main__.DataShell object at 0x7fd5c8458550>







### The Init Method

# Now it's time to explore the special __init__ method!

# __init__ is an initialization method used to construct class instances in custom ways. 
# In this exercise we will simply introduce the utilization of the method, and in 
# subsequent ones we will do fancier things.

# Create class: DataShell
class DataShell:
  
	# Initialize class with self argument
    def __init__(self):
      
        # Pass statement
        pass

# Instantiate DataShell: my_data_shell
my_data_shell = DataShell()

# Print my_data_shell
print(my_data_shell)

<script.py> output:
    <__main__.DataShell object at 0x7f3a8e4a3fd0>







### Instance Variables

# Class instances are useful in that we can store values in them at the time of 
# instantiation. We store these values in instance variables. This means that we 
# can have many instances of the same class whose instance variables hold different values!

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and integerInput arguments
    def __init__(self, integerInput):
      
		# Set data as instance variable, and assign the value of integerInput
        self.data = integerInput

# Declare variable x with value of 10
x = 10      

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell
print(my_data_shell.data)

<script.py> output:
    10

    # Great job declaring instance variables! Notice that instance variables live in the 
    # body of the initialization method, as they are initialized when the object is instantiated. 
    # Also important to notice that they are preceded by self., as this is referring to 
    # the instance itself.







### Multiple Instance Variables

# We are not limited to declaring only one instance variable; in fact, we can declare many!
# In this exercise we will declare two instance variables: identifier and data. 
# Their values will be specified by the values passed to the initialization method, 
# as before.

# Create class: DataShell
class DataShell:
  
	# Initialize class with self, identifier and data arguments
    def __init__(self, identifier, data):
      
		# Set identifier and data as instance variables, assigning value of input arguments
        self.identifier = identifier
        self.data = data

# Declare variable x with value of 100, and y with list of integers from 1 to 5
x = 100
y = [1, 2, 3, 4, 5]

# Instantiate DataShell passing x and y as arguments: my_data_shell
my_data_shell = DataShell(x,y)

# Print my_data_shell.identifier
print(my_data_shell.identifier)

# Print my_data_shell.data
print(my_data_shell.data)

<script.py> output:
    100
    [1, 2, 3, 4, 5]







### Class Variables

# We saw that we can specify different instance variables.
# But, what if we want any instance of a class to hold the same value for a specific 
# variable? Enter class variables.
# Class variables must not be specified at the time of instantiation and instead, 
# are declared/specified at the class definition phase.

# Create class: DataShell
class DataShell:
  
    # Declare a class variable family, and assign value of "DataShell"
    family = "DataShell"
    
    # Initialize class with self, identifier arguments
    def __init__(self, identifier):
      
        # Set identifier as instance variable of input argument
        self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x as argument: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)

<script.py> output:
    DataShell

    # Awesome! Class variables are different from instance variables (which we saw earlier). 
    # Even though class variables may be overridden, they are generally set even before object 
    # instanciation; therefore, class variable values across instances of the same class tend 
    # to be the same.







### Overriding Class Variables

# Sometimes our object instances have class variables whose values are not correct, 
# and hence, not useful. For this reason it makes sense to modify our object's class variables.
# In this exercise, we will do just that: override class variables with values of our own!

# Create class: DataShell
class DataShell:
  
    # Declare a class variable family, and assign value of "DataShell"
    family = "DataShell"
    
    # Initialize class with self, identifier arguments
    def __init__(self, identifier):
      
        # Set identifier as instance variables, assigning value of input arguments
        self.identifier = identifier

# Declare variable x with value of 100
x = 100

# Instantiate DataShell passing x and y as arguments: my_data_shell
my_data_shell = DataShell(x)

# Print my_data_shell class variable family
print(my_data_shell.family)

# Override the my_data_shell.family value with "NotDataShell"
my_data_shell.family = "NotDataShell"

# Print my_data_shell class variable family once again
print(my_data_shell.family)

<script.py> output:
    DataShell
    NotDataShell







### Methods I

# Not only are we able to declare both instance variables and class variables in our 
# objects, we can also cook functions right into our objects as well. These object-contained 
# functions are called methods.

# Create class: DataShell
class DataShell:
  
	# Initialize class with self argument
    def __init__(self):
        pass
      
	# Define class method which takes self argument: print_static
    def print_static(self):
        # Print string
        print("You just executed a class method!")
        
# Instantiate DataShell taking no arguments: my_data_shell
my_data_shell = DataShell()

# Call the print_static method of your newly created object
my_data_shell.print_static()

<script.py> output:
    You just executed a class method!







### Methods II

# In the previous exercise our print_static() method was kind of boring. We can do more 
# interesting things with our objects' methods. For example, we can interact with our objects' 
# data. In this exercise we will declare a method that prints the value of one of our instance 
# variables.

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define class method which takes self argument: show
    def show(self):
        # Print the instance variable data
        print(self.data)

# Declare variable with list of integers from 1 to 10: integer_list   
integer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show method of your newly created object
my_data_shell.show()

<script.py> output:
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]







### Methods III

# In the last exercise our method simply printed out the value of instance variables.
# In this one, we'll do something more interesting. We will add another method, avg(), 
# which takes a list of integers, calculates the average value, and prints it out. 
# To make things even more interesting, the list of integers for which avg() does this 
# operations, is one of our object's instance variables.

# This means that our object can not only store data, but also can store procedures 
# it can execute on its own data. Awesome. Note that the variable integer_list has already 
# been loaded for you.

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define method that prints data: show
    def show(self):
        print(self.data)
        
    # Define method that prints average of data: avg 
    def avg(self):
        # Declare avg and assign it the average of data
        avg = sum(self.data)/float(len(self.data))
        # Print avg
        print(avg)
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Call the show and avg methods of your newly created object
my_data_shell.show()
my_data_shell.avg()

<script.py> output:
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    5.5















######## Chapter 3: Fancy classes, fancy objects















### Return Statement I

# Let's now drill into the return statement.

# class DataShell:
#     def __init__(self, x):
#         return x
# In the code chunk above, you may have expected to see the print() function instead 
# of the return statement. The difference between the two is that print() outputs a 
# string to the console, while the the return statement exits the current function 
# (or method) and hands the returned value back to its caller. In this case, the caller 
# could have another function, among other things. If this sounds confusing have not fear, 
# we will further practice this!

# In the console, enter this code in order to answer the question below:

# x = my_data_shell.get_data()
# print(x)
# What value does the my_data_shell.get_data() method return?

In [2]: print(x)
[1, 2, 3, 4, 5]







### Return Statement II: The Return of the DataShell

# Let's now go back to the class DataShell that we were working with earlier, and 
# refactor it such that it uses the return statement instead of the print() function.

# Notice that since we are now using the return statement, we need to include our 
# calls to object methods within the print() function.

# Create class: DataShell
class DataShell:
  
	# Initialize class with self and dataList as arguments
    def __init__(self, dataList):
      	# Set data as instance variable, and assign it the value of dataList
        self.data = dataList
        
	# Define method that returns data: show
    def show(self):
        return self.data
        
    # Define method that returns average of data: avg 
    def avg(self):
        # Declare avg and assign it the average of data
        avg = sum(self.data)/float(len(self.data))
        # Return avg
        return avg
        
# Instantiate DataShell taking integer_list as argument: my_data_shell
my_data_shell = DataShell(integer_list)

# Print output of your object's show method
print(my_data_shell.show())

# Print output of your object's avg method
print(my_data_shell.avg())

<script.py> output:
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    5.5







### Return Statement III: A More Powerful DataShell

# In this exercise our DataShell class will evolve from simply consuming lists to consuming 
# CSV files so that we can do more sophisticated things.

# For this, we will employ the return statement once again. Additionally, we will 
# leverage some neat functionality from both the numpy and pandas packages.

# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
  
    # Initialize class with self and inputFile
    def __init__(self, inputFile):
        self.file = inputFile
        
    # Define generate_csv method, with self argument
    def generate_csv(self):
        self.data_as_csv = pd.read_csv(self.file)
        return self.data_as_csv

# Instantiate DataShell with us_life_expectancy as input argument
data_shell = DataShell(us_life_expectancy)

# Call data_shell's generate_csv method, assign it to df
df = data_shell.generate_csv()

# Print df
print(df)

<script.py> output:
               country       ...       life_expectancy
    0    United States       ...             39.410000
    1    United States       ...             45.209999
    2    United States       ...             49.299999
    3    United States       ...             50.500000







### Data as Attribute

# In the previous coding exercise you wrote a method within your DataShell class that 
# returns a Pandas Dataframe.In this one, we will cook the data into our class, as an instance variable. 
# This is so that we can do fancy things later, such as renaming columns, as well as getting 
# descriptive statistics. The object us_life_expectancy is loaded and available in your workspace.

# Import numpy as np, pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
  
    # Define initialization method
    def __init__(self, filepath):
        # Set filepath as instance variable  
        self.filepath = filepath
        # Set data_as_csv as instance variable
        self.data_as_csv = pd.read_csv(filepath)

# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print your object's data_as_csv attribute
print(us_data_shell.data_as_csv)

<script.py> output:
               country code  year  life_expectancy
    0    United States  USA  1880        39.410000
    1    United States  USA  1890        45.209999
    2    United States  USA  1901        49.299999
    3    United States  USA  1902        50.500000...







### Renaming Columns

# Methods can be especially useful to manipulate their object's data. In this 
# exercise, we will create a method inside of our DataShell class, so that we 
# can rename our data columns. numpy and pandas are already imported in your workspace 
# as np and pd, respectively.

# Create class DataShell
class DataShell:
  
    # Define initialization method
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)
    
    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)

# Instantiate DataShell as us_data_shell with argument us_life_expectancy
us_data_shell = DataShell(us_life_expectancy)

# Print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

# Rename your objects column 'code' to 'country_code'
us_data_shell.rename_column('code', 'country_code')

# Again, print the datatype of your object's data_as_csv attribute
print(us_data_shell.data_as_csv.dtypes)

<script.py> output:
    country             object
    code                object
    year                 int64
    life_expectancy    float64
    dtype: object
    country             object
    country_code        object
    year                 int64
    life_expectancy    float64
    dtype: object







### Self-Describing DataShells

# In this exercise you will add functionality to your DataShell class such that it 
# returns information about itself. numpy and pandas are already imported in your 
# workspace as np and pd, respectively.

# Create class DataShell
class DataShell:

    # Define initialization method
    def __init__(self, filepath):
        self.filepath = filepath
        self.data_as_csv = pd.read_csv(filepath)

    # Define method rename_column, with arguments self, column_name, and new_column_name
    def rename_column(self, column_name, new_column_name):
        self.data_as_csv.columns = self.data_as_csv.columns.str.replace(column_name, new_column_name)
        
    # Define get_stats method, with argument self
    def get_stats(self):
        # Return a description data_as_csv
        return self.data_as_csv.describe()
    
# Instantiate DataShell as us_data_shell
us_data_shell = DataShell(us_life_expectancy)

# Print the output of your objects get_stats method
print(us_data_shell.get_stats())

<script.py> output:
                  year  life_expectancy
    count   117.000000       117.000000
    mean   1956.752137        66.556684
    std      34.398252         9.551079
    min    1880.000000        39.410000
    25%    1928.000000        58.500000
    50%    1957.000000        69.599998
    75%    1986.000000        74.772003
    max    2015.000000        79.244003















######## Chapter 4: Inheritance, polymorphism and composition

















### Animal Inheritance

# In this exercise we will code a simple example of an abstract class, and two other 
# classes that inherit from it. To focus on the concept of inheritance, we will introduce 
# another set of classes: Animal, Mammal, and Reptile. More specifically, Animal 
# will be our abstract class, and both Mammal and Reptile 
# will inherit from it.

# Create a class Animal
class Animal:
	def __init__(self, name):
		self.name = name

# Create a class Mammal, which inherits from Animal
class Mammal(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Create a class Reptile, which also inherits from Animal
class Reptile(Animal):
	def __init__(self, name, animal_type):
		self.animal_type = animal_type

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print both objects
print(daisy)
print(stella)

<script.py> output:
    <__main__.Mammal object at 0x7f82f886d860>
    <__main__.Reptile object at 0x7f82f886d978>







### Vertebrate Inheritance

# In the previous exercise, it seemed almost unnecessary to have an abstract class, 
# as it did not do anything particularly interesting (other than begin to learn inheritance).
# In this exercise, we will refine our abstract class and include some class variables 
# in our abstract class so that they can be passed down to our other classes.
# Additionally from inheritance, in this exercise we are seeing another powerful 
# object-oriented programming concept: polymorphism. As you explore your code while 
# writing the Mammal and Reptile classes, notice their differences. Because they both 
# inherit from the Vertebrate class, and because they are different, we say that they 
# are polymorphic. How cool!

# Create a class Vertebrate
class Vertebrate:
    spinal_cord = True
    def __init__(self, name):
        self.name = name

# Create a class Mammal, which inherits from Vertebrate
class Mammal(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = True

# Create a class Reptile, which also inherits from Vertebrate
class Reptile(Vertebrate):
    def __init__(self, name, animal_type):
        self.animal_type = animal_type
        self.temperature_regulation = False

# Instantiate a mammal with name 'Daisy' and animal_type 'dog': daisy
daisy = Mammal('Daisy', 'dog')

# Instantiate a reptile with name 'Stella' and animal_type 'alligator': stella
stella = Reptile('Stella', 'alligator')

# Print stella's attributes spinal_cord and temperature_regulation
print("Stella Spinal cord: " + str(stella.spinal_cord))
print("Stella temperature regularization: " + str(stella.temperature_regulation))

# Print daisy's attributes spinal_cord and temperature_regulation
print("Daisy Spinal cord: " + str(daisy.spinal_cord))
print("Daisy temperature regularization: " + str(daisy.temperature_regulation))

<script.py> output:
    Stella Spinal cord: True
    Stella temperature regularization: False
    Daisy Spinal cord: True
    Daisy temperature regularization: True







### Abstract Class DataShell I

# We will now switch back to working on our DataShell class. Specifically, we will 
# create an abstract class, such that we can create other classes that then inherit 
# from it! For this reason, our abstract DataShell class will not do much, resembling 
# some of the earlier exercises in this course.

# Load numpy as np and pandas as pd
import numpy as np
import pandas as pd

# Create class: DataShell
class DataShell:
    def __init__(self, inputFile):
        self.file = inputFile

# Instantiate DataShell as my_data_shell
my_data_shell = DataShell(us_life_expectancy)

# Print my_data_shell
print(my_data_shell)

<script.py> output:
    <__main__.DataShell object at 0x7f82e57b2b70>







### Abstract Class DataShell II

# Now that we have our abstract class DataShell, we can now create a second class 
# that inherits from it. Specifically, we will define a class called CsvDataShell. 
# This class will have the ability to import a CSV file. In the following exercises 
# we will add a bit more functionality to make our classes more sophisticated!






