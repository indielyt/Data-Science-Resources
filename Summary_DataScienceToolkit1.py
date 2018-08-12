# Datacamp INTRODUCTION TO TIME SERIES ANALYSIS IN PYTHON

# Chapter 1: Writing your own functions
	
		### Write a simple function
		### Single-parameter functions
		### Functions that return single values
		### Functions with multiple parameters
		### A brief introduction to tuples
		### Functions that return multiple values
		### Bringing it all together (1)
		### Bringing it all together (2)


# Chapter 2: Default arguments, variable-length arguments and scope
	
		### The keyword global
		### Python's built-in scope
		### Nested Functions I
		### Nested Functions II
		### The keyword nonlocal and nested functions
		### Functions with one default argument
		### Functions with multiple default arguments
		### Functions with variable-length arguments (*args)
		### Functions with variable-length keyword arguments (**kwargs)
		### Bringing it all together (1)


# Chapter 3: 
	
		### Writing a lambda function you already know
		### Map() and lambda functions
		### Filter() and lambda functions
		### Reduce() and lambda functions
		### Error handling with try-except
		### Error handling by raising an error
		### Bringing it all together (1)
		### Bringing it all together (2)
















################ Chapter 1: Writing your own functions
















### Write a simple function

# Define the function shout
def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = 'congratulations' + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout
shout()







### Single-parameter functions

# Define shout with the parameter, word
def shout(word):
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Print shout_word
    print(shout_word)

# Call shout with the string 'congratulations'
shout('congratulations')







### Functions that return single values

# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Replace print with return
    return shout_word

# Pass 'congratulations' to shout: yell
yell = shout('congratulations')

# Print yell
print (yell)







### Functions with multiple parameters

# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1 + shout2

    # Return new_shout
    return new_shout

# Pass 'congratulations' and 'you' to shout(): yell
yell = shout('congratulations', 'you')

# Print yell
print(yell)







### A brief introduction to tuples

In [1]: print (nums)
(3, 4, 6)

# Unpack nums into num1, num2, and num3
num1, num2, num3 = nums

# Construct even_nums
even_nums = (2, num2, num3)







### Functions that return multiple values

# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)








### Bringing it all together (1)

# For this exercise, your goal is to recall how to load a dataset into a DataFrame. 
# The dataset contains Twitter data and you will iterate over entries in a column to 
# build a dictionary in which the keys are the names of languages and the values 
# are the number of tweets in the given language. The file tweets.csv is 
# available in your current directory.

# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)







### Bringing it all together (2)

# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df, 'lang')

# Print the result
print(result)















### Chapter 2: Default arguments, variable-length arguments and scope















### The keyword global

# Let's work more on your mastery of scope. In this exercise, you will use the 
# keyword global within a function to alter the value of a variable defined in 
# the global scope.

# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():
    """Change the value of the global variable team."""

    # Use team in global scope
    global team

    # Change the value of team in global: team
    team = 'justice league'

# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)







### Python's built-in scope

# Here you're going to check out Python's built-in scope, which is really just a 
# built-in module called builtins. However, to query builtins, you'll need to 
# import builtins 'because the name builtins is not itself built in...No, I’m 
# serious!' (Learning Python, 5th edition, Mark Lutz). After executing import 
# builtins in the IPython Shell, execute dir(builtins) to print a list of all 
# the names in the module builtins. Have a look and you'll see a bunch of names 
# that you'll recognize! Which of the following names is NOT in the 
# module builtins?

In [1]: import builtins

In [2]: dir(builtins)
Out[2]: 
['ArithmeticError',
 'AssertionError',
 'AttributeError',
 'BaseException',
 'BlockingIOError',
 'BrokenPipeError',
 'BufferError',
 'BytesWarning',
 'ChildProcessError',
 'ConnectionAbortedError',
 'ConnectionError',
 'ConnectionRefusedError',
 'ConnectionResetError',
 'DeprecationWarning',
 'EOFError',
 'Ellipsis',
 'EnvironmentError',
 'Exception',
 'False',
 'FileExistsError',
 'FileNotFoundError',
 'FloatingPointError',
 'FutureWarning',
 'GeneratorExit',
 'IOError',
 'ImportError',
 'ImportWarning',
 'IndentationError',
 'IndexError',
 'InterruptedError',
 'IsADirectoryError',
 'KeyError',
 'KeyboardInterrupt',
 'LookupError',
 'MemoryError',
 'NameError',
 'None',
 'NotADirectoryError',
 'NotImplemented',
 'NotImplementedError',
 'OSError',
 'OverflowError',
 'PendingDeprecationWarning',
 'PermissionError',
 'ProcessLookupError',
 'RecursionError',
 'ReferenceError',
 'ResourceWarning',
 'RuntimeError',
 'RuntimeWarning',
 'StopAsyncIteration',
 'StopIteration',
 'SyntaxError',
 'SyntaxWarning',
 'SystemError',
 'SystemExit',
 'TabError',
 'TimeoutError',
 'True',
 'TypeError',
 'UnboundLocalError',
 'UnicodeDecodeError',
 'UnicodeEncodeError',
 'UnicodeError',
 'UnicodeTranslateError',
 'UnicodeWarning',
 'UserWarning',
 'ValueError',
 'Warning',
 'ZeroDivisionError',
 '_',
 '__IPYTHON__',
 '__IPYTHON__active',
 '__build_class__',
 '__debug__',
 '__doc__',
 '__import__',
 '__loader__',
 '__name__',
 '__package__',
 '__spec__',
 'abs',
 'all',
 'any',
 'ascii',
 'bin',
 'bool',
 'bytearray',
 'bytes',
 'callable',
 'chr',
 'classmethod',
 'compile',
 'complex',
 'copyright',
 'credits',
 'delattr',
 'dict',
 'dir',
 'divmod',
 'dreload',
 'enumerate',
 'eval',
 'exec',
 'filter',
 'float',
 'format',
 'frozenset',
 'get_ipython',
 'getattr',
 'globals',
 'hasattr',
 'hash',
 'help',
 'hex',
 'id',
 'input',
 'int',
 'isinstance',
 'issubclass',
 'iter',
 'len',
 'license',
 'list',
 'locals',
 'map',
 'max',
 'memoryview',
 'min',
 'next',
 'object',
 'oct',
 'open',
 'ord',
 'pow',
 'print',
 'property',
 'range',
 'repr',
 'reversed',
 'round',
 'set',
 'setattr',
 'slice',
 'sorted',
 'staticmethod',
 'str',
 'sum',
 'super',
 'tuple',
 'type',
 'vars',
 'zip']







 ### Nested Functions I

 # Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))

<script.py> output:
    ('a!!!', 'b!!!', 'c!!!')







### Nested Functions II

# Great job, you've just nested a function within another function. One other 
# pretty cool reason for nesting functions is the idea of a closure. This means 
# that the nested or inner function remembers the state of its enclosing scope 
# when called. Thus, anything defined locally in the enclosing scope is available 
# to the inner function even when the outer function has finished execution.

# Let's move forward then! In this exercise, you will complete the definition of 
# the inner function inner_echo() and then call echo() a couple of times, each 
# with a different argument. Complete the exercise and see what the output will be!

# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))







### The keyword nonlocal and nested functions

# Let's once again work further on your mastery of scope! In this exercise, 
# you will use the keyword nonlocal within a nested function to alter the 
# value of a variable defined in the enclosing scope.

# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word = word*2
    
    # Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        # Use echo_word in nonlocal scope
        nonlocal echo_word
        
        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + '!!!'
    
    # Call function shout()
    shout()
    
    # Print echo_word
    print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout('hello')

<script.py> output:
    hellohello
    hellohello!!!







### Functions with one default argument

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
     exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo('Hey')

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo('Hey', echo=5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)

<script.py> output:
    Hey!!!
    HeyHeyHeyHeyHey!!!







### Functions with multiple default arguments

# You've now defined a function that uses a default argument - don't stop there 
# just yet! You will now try your hand at defining a function with more than 
# one default argument and then calling this function in various ways.

# After defining the function, you will call it by supplying values to all 
# the default arguments of the function. Additionally, you will call the function 
# by not passing a value to one of the default arguments - see how that changes 
# the output of your function!

# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Capitalize echo_word if intense is True
    if intense is True:
        # Capitalize and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo('Hey', echo=5, intense=True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo('Hey', intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)

<script.py> output:
    HEYHEYHEYHEYHEY!!!
    HEY!!!







### Functions with variable-length arguments (*args)

# Flexible arguments enable you to pass a variable number of arguments to a 
# function. In this exercise, you will practice defining a function that accepts 
# a variable number of string arguments.

# The function you will define is gibberish() which can accept a variable number 
# of string values. Its return value is a single string composed of all the 
# string arguments concatenated together in the order they were passed to the 
# function call. You will call the function with a single string argument and 
# see how the output changes with another call using more than one string argument. 
# Recall from the previous video that, within the function definition, args is a tuple.

# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ""

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)







### Functions with variable-length keyword arguments (**kwargs)

# Let's push further on what you've learned about flexible arguments - you've 
# used *args, you're now going to use **kwargs! What makes **kwargs different 
# is that it allows you to pass a variable number of keyword arguments to 
# functions. Recall from the previous video that, within the function 
# definition, kwargs is a dictionary.

# To understand this idea better, you're going to use **kwargs in this exercise 
# to define a function that accepts a variable number of keyword arguments. 
# The function simulates a simple status report system that prints out the 
# status of a character in a movie.

# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name="luke", affiliation="jedi", status="missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")

<script.py> output:
    
    BEGIN: REPORT
    
    status: missing
    affiliation: jedi
    name: luke
    
    END REPORT
    
    BEGIN: REPORT
    
    status: deceased
    affiliation: sith lord
    name: anakin
    
    END REPORT







 ### Bringing it all together (1)

#  Recall the Bringing it all together exercise in the previous chapter where 
#  you did a simple Twitter analysis by developing a function that counts how 
#  many tweets are in certain languages. The output of your function was a 
#  dictionary that had the language as the keys and the counts of tweets in 
#  that language as the value.

# In this exercise, we will generalize the Twitter language analysis that you 
# did in the previous chapter. You will do that by including a default argument 
# that takes a column name.

# For your convenience, pandas has been imported as pd and the 'tweets.csv' 
# file has been imported into the DataFrame tweets_df. Parts of the code from 
# your previous work are also provided.

# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'source')

# Print result1 and result2
print(result1)
print(result2)







### Bringing it all together (2)

# Wow, you've just generalized your Twitter language analysis that you did in 
# the previous chapter to include a default argument for the column name. You're 
# now going to generalize this function one step further by allowing the user to 
# pass it a flexible argument, that is, in this case, as many column names as 
# the user would like!

# Once again, for your convenience, pandas has been imported as pd and the 
# 'tweets.csv' file has been imported into the DataFrame tweets_df. Parts of 
# the code from your previous work are also provided.

# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)















############### Chapter 3: Lambda functions and error-handling














### Writing a lambda function you already know

# Some function definitions are simple enough that they can be converted to a 
# lambda function. By doing this, you write less lines of code, which is pretty 
# awesome and will come in handy, especially when you're writing and maintaining 
# big programs. In this exercise, you will use what you know about lambda 
# functions to convert a function that does a simple task into a lambda 
# function. Take a look at this function definition:

def echo_word(word1, echo):
    """Concatenate echo copies of word1."""
    words = word1 * echo
    return words

# The function echo_word takes 2 parameters: a string value, word1 and an 
# integer value, echo. It returns a string that is a concatenation of echo 
# copies of word1. Your task is to convert this simple function 
# into a lambda function.

# Define echo_word as a lambda function: echo_word
echo_word = lambda word1, echo: word1*echo

# Call echo_word: result
result = echo_word('hey', 5)

# Print result
print(result)

<script.py> output:
    heyheyheyheyhey







 ### Map() and lambda functions

# Note: Map() combines the lambda function, and lambda function call into 
# one line.

#  So far, you've used lambda functions to write short, simple functions as well 
#  as to redefine functions with simple functionality. The best use case for 
#  lambda functions, however, are for when you want these simple functionalities 
#  to be anonymously embedded within larger expressions. What that means is 
#  that the functionality is not stored in the environment, unlike a function 
#  defined with def. To understand this idea better, you will use a lambda 
#  function in the context of the map() function.

# Recall from the video that map() applies a function over an object, such as a 
# list. Here, you can use lambda functions to define the function that map() 
# will use to process the object. For example:

nums = [2, 4, 6, 8, 10]
result = map(lambda a: a ** 2, nums)

# You can see here that a lambda function, which raises a value a to the power 
# of 2, is passed to map() alongside a list of numbers, nums. The map object 
# that results from the call to map() is stored in result. You will now practice 
# the use of lambda functions with map(). For this exercise, you will map the 
# functionality of the add_bangs() function you defined in previous exercises 
# over a list of strings.

# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda item: item + '!!!' , spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Convert shout_spells into a list and print it
print(shout_spells_list)

<script.py> output:
    ['protego!!!', 'accio!!!', 'expecto patronum!!!', 'legilimens!!!']








### Filter() and lambda functions

# In the previous exercise, you used lambda functions to anonymously embed an 
# operation within map(). You will practice this again in this exercise 
# by using a lambda function with filter(), which may be new to you! The 
# function filter() offers a way to filter out elements from a list that don't 
# satisfy certain criteria.

# Your goal in this exercise is to use filter() to create, from an input 
# list of strings, a new list that contains only strings that have more than 
# 6 characters.

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6 , fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Convert result into a list and print it
print(result_list)

# Output:
['samwise', 'aragorn', 'boromir', 'legolas', 'gandalf']







### Reduce() and lambda functions

# You're getting very good at using lambda functions! Here's one more function 
# to add to your repertoire of skills. The reduce() function is useful for 
# performing some computation on a list and, unlike map() and filter(), returns 
# a single value as a result. To use reduce(), you must import it from the 
# functools module.  Remember gibberish() from a few exercises back?

# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""
    hodgepodge = ''
    for word in args:
        hodgepodge += word
    return hodgepodge

# gibberish() simply takes a list of strings as an argument and returns, as a 
# single-value result, the concatenation of all of these strings. In this 
# exercise, you will replicate this functionality by using reduce() and a 
# lambda function that concatenates strings together.

# Import reduce from functools
from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1+item2, stark)

# Print the result
print(result)

<script.py> output:
    robbsansaaryabrandonrickon







 ### Error handling with try-except

# A good practice in writing your own functions is also anticipating the ways in 
# which other people (or yourself, if you accidentally misuse your own function) 
# might use the function you defined.

# As in the previous exercise, you saw that the len() function is able to handle 
# input arguments such as strings, lists, and tuples, but not int type ones and 
# raises an appropriate error and error message when it encounters invalid input 
# arguments. One way of doing this is through exception handling with 
# the try-except block.

# In this exercise, you will define a function as well as use a try-except 
# block for handling cases when incorrect input arguments are passed to the function.

# Recall the shout_echo() function you defined in previous exercises; parts 
# of the function definition are provided in the sample code. Your goal is to 
# complete the exception handling code in the function definition and provide 
# an appropriate error message when raising an error.

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Initialize empty strings: echo_word, shout_words
    echo_word, shout_words = [], []
    

    # Add exception handling with try-except
    try:
        # Concatenate echo copies of word1 using *: echo_word
        echo_word = word1*echo

        # Concatenate '!!!' to echo_word: shout_words
        shout_words = echo_word + '!!!'
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")

    # Return shout_words
    return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")

<script.py> output:
    word1 must be a string and echo must be an integer.







### Error handling by raising an error

# Another way to raise an error is by using raise. In this exercise, you will 
# add a raise statement to the shout_echo() function you defined before to 
# raise an error message when the value supplied by the user to the echo 
# argument is less than 0.

# The call to shout_echo() uses valid argument values. To test and see how 
# the raise statement works, simply change the value for the echo argument 
# to a negative value. Don't forget to change it back to valid values to move 
# on to the next exercise!

# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three
    exclamation marks at the end of the string."""

    # Raise an error with raise
    if echo<0:
        raise ValueError('echo must be greater than 0')

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word

# Call shout_echo
shout_echo("particle", echo=5)

Out[1]: 'particleparticleparticleparticleparticle!!!'







### Bringing it all together (1)

# This is awesome! You have now learned how to write anonymous functions using 
# lambda, how to pass lambda functions as arguments to other functions such as 
# map(), filter(), and reduce(), as well as how to write errors and output custom 
# error messages within your functions. You will now put together these learnings 
# to good use by working with a Twitter dataset. Before practicing your new error 
# handling skills,in this exercise, you will write a lambda function and use 
# filter() to select retweets, that is, tweets that begin with the string 'RT'.

# To help you accomplish this, the Twitter data has been imported into the 
# DataFrame, tweets_df. Go for it!

# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: x[0:2]=='RT', tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)

# Output:
   RT @TUSK81: LOUDER FOR THE PEOPLE IN THE BACK https://t.co/hlPVyNLXzx
    RT @loopzoop: Well...put it back https://t.co/8Yb7BDT5VM








### Bringing it all together (2)

# Sometimes, we make mistakes when calling functions - even ones you made 
# yourself. But don't fret! In this exercise, you will improve on your previous 
# work with the count_entries() function in the last chapter by adding a 
# try-except block to it. This will allow your function to provide a helpful 
# message when the user calls your count_entries() function but provides a 
# column name that isn't in the DataFrame.

# Once again, for your convenience, pandas has been imported as pd and the 
# 'tweets.csv' file has been imported into the DataFrame tweets_df. Parts of 
# the code from your previous work are also provided.

# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Add try block
    try:
        # Extract column from DataFrame: col
        col = df[col_name]
        
        # Iterate over the column in dataframe
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1
    
        # Return the cols_count dictionary
        return cols_count

    # Add except block
    except:
        'The DataFrame does not have a ' + col_name + ' column.'

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)







### Bringing it all together (3)

# In the previous exercise, you built on your function count_entries() to add a 
# try-except block. This was so that users would get helpful messages when 
# calling your count_entries() function and providing a column name that isn't 
# in the DataFrame. In this exercise, you'll instead raise a ValueError in the 
# case that the user provides a column name that isn't in the DataFrame.

# Once again, for your convenience, pandas has been imported as pd and the 
# 'tweets.csv' file has been imported into the DataFrame tweets_df. Parts of 
# the code from your previous work are also provided.

# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError ('The DataFrame does not have a ' + col_name + ' column.')

    # Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1
        
        # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, col_name = 'lang')

# Print result1
print(result1)







