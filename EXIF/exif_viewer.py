# dependencies
import os
import re

from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngImageFile, PngInfo


def read_png(image_path):

    exif_array=[]
    type = Image.open(image_path)

    image = PngImageFile(image_path)
    metadata = PngInfo()

    # Compile array from tags dict
    for i in image.text:
        compile = i, str(image.text[i])
        exif_array.append(compile)
      # If XML metadata, pull out data by idenifying data type and gathering useful meta
    if len(exif_array) > 0:
        header = exif_array[0][0]
    else:
        header = ""
        print("No available metadata")

    xml_output = []
    if header.startswith("XML"):
      xml = exif_array[0][1]
      xml_output.extend(xml.splitlines())
      # Remove useless meta tags
      for line in xml.splitlines():
         if "<" not in line:
           if "xmlns" not in line:
             # Remove equal signs, quotation marks, /> characters and leading spaces
             xml_line = re.sub(r'[a-z]*:', '', line).replace('="', ': ')
             xml_line = xml_line.rstrip(' />')
             xml_line = xml_line.rstrip('\"')
             xml_line = xml_line.lstrip(' ')
             print(xml_line)
           elif header.startswith("Software"):
              print("No available metadata")

            # If no XML, print available metadata
         else:
            for properties in exif_array:
                if properties[0] != 'JPEGThumbnail':
                    print(': '.join(str(x) for x in properties))

def read_png2(image_path):
    im = Image.open(image_path)
    im.load()  # Needed only for .png EXIF data (see citation above)

    if len(im.info['exif']) == 0:
        print('This image has no exif data.')
    else:
        print(im.info['exif'])


def exif_viewer(image_path: str):

    # this will return a tuple of root and extension
    image_file = image_path.split('/')[-1]
    split_tup = os.path.splitext(image_file)
    # extract the file name and extension
    file_name = split_tup[0]
    file_extension = split_tup[1]
    if file_extension not in ['.png','.jpg']:
        raise TypeError('Please make sure you type down the file extension, and only .png and .jpg accepted')

    if file_extension == '.png':
        read_png2(image_path)

    else:
        img = Image.open(image_path)
        img_exif = img.getexif()

        if len(img_exif.items()) == 0:
            message = 'This image has no exif data.'
            return message
        else:
            for key, val in img_exif.items():
                if key in ExifTags.TAGS:
                    message = 'EXIF information of the image.   '+f'{ExifTags.TAGS[key]}:{val}'
                    return message