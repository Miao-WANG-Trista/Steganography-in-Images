# define a function to convert any type of data into binary
import numpy as np


def messagetobinary(message):
    if type(message) == str:
        return ''.join([format(ord(i), "08b") for i in message])
        # ord() to convert a string into a integer(ASCII value) and format(,"08b") is to convert a integer into a binary value
    elif type(message) == bytes or type(message) == np.ndarray:
        # bytes or np.ndarray stores a list/array of numerical elements
        return [format(i, "08b") for i in message]
    elif type(message) == int or type(message) == np.uint8:
        return format(message, "08b")
    else:
        raise TypeError("Input Type Not Supported")