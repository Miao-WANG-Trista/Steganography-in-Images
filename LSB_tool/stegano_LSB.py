# import all the required packages
import cv2
# cv2.startWindowThread()
import numpy as np
import types
from LSB_tool.convert_to_binary import messagetobinary
from PIL import Image
import os
from pysteganography.stegano import decode
def converter(image_path):
    image_file = image_path.split('/')[-1]
    split_tup = os.path.splitext(image_file)
    # extract the file name and extension
    file_name = split_tup[0]
    file_extension = split_tup[1]
    if file_extension not in ['.png','.jpg']:
        raise TypeError('Please make sure you type down the file extension, and only .png and .jpg accepted')

    if file_extension == '.jpg':
        im = Image.open(image_path)
        current_dir = os.getcwd()
        file_name = file_name+'steg.png'
        image_path = os.path.join(current_dir,file_name)
        im.save(image_path)
    return image_path

def encoder(image_path, secret_message):  # this will return a tuple of root and extension
    # calculate the maximum bytes to encode
  image = cv2.imread(image_path)
  n_bytes = image.shape[0] * image.shape[1] * 3 // 8
  print("Maximum bytes to encode:", n_bytes)

  #Check if the number of bytes to encode is less than the maximum bytes in the image
  if len(secret_message) > n_bytes:
      raise ValueError("Error encountered insufficient bytes, need bigger image or less data !!")

  secret_message += "#####" # you can use any string as the delimeter

  data_index = 0
  # convert input data to binary format using messageToBinary() fucntion
  binary_secret_msg = messagetobinary(secret_message)

  data_len = len(binary_secret_msg) #Find the length of data that needs to be hidden
  for values in image:
      for pixel in values:
          # convert RGB values to binary format
          r, g, b = messagetobinary(pixel)
          # modify the least significant bit only if there is still data to store
          if data_index < data_len:
              # hide the data into least significant bit of red pixel
              pixel[0] = int(r[:-1] + binary_secret_msg[data_index], 2)
              data_index += 1
          if data_index < data_len:
              # hide the data into least significant bit of green pixel
              pixel[1] = int(g[:-1] + binary_secret_msg[data_index], 2)
              data_index += 1
          if data_index < data_len:
              # hide the data into least significant bit of  blue pixel
              pixel[2] = int(b[:-1] + binary_secret_msg[data_index], 2)
              data_index += 1
          # if data is encoded, just break out of the loop
          if data_index >= data_len:
              break

  return image
# Decoder function to reveal the hidden message in a LSB-based stegno image

def decoder(image_path):
    image_path=converter(image_path)
    image = cv2.imread(image_path)
    binary_data = ""
    for values in image:
          for pixel in values:
              r, g, b = messagetobinary(pixel) #convert the red,green and blue values into binary format
              binary_data += r[-1] #extracting data from the least significant bit of red pixel
              binary_data += g[-1] #extracting data from the least significant bit of red pixel
              binary_data += b[-1] #extracting data from the least significant bit of red pixel
    # split by 8-bits
    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "#####": #check if we have reached the delimeter which is "#####"
            break
    #print(decoded_data)
    return decoded_data[:-5] #remove the delimeter to show the original hidden message

# Function to take the input and call the encoder function
def encode_text():
    image_path = input("PLease enter the image name: including the extension")
    image = cv2.imread(image_path)

    # print details of the image
    print("the shape of the image is: ", image.shape)
    print("the original image is shown as below:")
    resized_image = cv2.resize(image,(800,300))
    # cv2.imshow('image', resized_image)
    # cv2.waitKey(0) # display until clicking on the image and press 'esc'
    # cv2.destroyAllWindows()

    data = input("Please enter a sentence that to be encoded")
    if len(data) == 0:
        raise ValueError("Data is empty")
    filename = input("Please enter a name for the stegno image")
    encoded_image = encoder(image_path,data)
    cv2.imwrite(filename,encoded_image) # save the image to the designated filename

# function to decode the image
def decode_text(encoded_image_path):
    image_file = encoded_image_path.split('/')[-1]
    split_tup = os.path.splitext(image_file)
    # extract the file name and extension
    file_name = split_tup[0]
    file_extension = split_tup[1]
    if file_extension not in ['.png']:
        raise TypeError('Please make sure you type down the file extension, and only .png accepted')

    text = decoder(encoded_image_path)

    return text


# Main function
if __name__ == "__main__":
    choice = int(input ("Image steganography \n 1. Encode the data \n 2. Decode the data \n Your input is: "))
    if choice == 1:
        print("Encoding....Pls click on the image and press 'esc' to exit")
        encode_text()
    elif choice == 2:
        print("\nDecoding......")
        encoded_image = input("Please enter the name of stegano image that you want to decode")
        print("Decoded message is " + decode_text(encoded_image))
    else:
        raise Exception("Please enter correct number")