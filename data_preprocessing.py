# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:15:33 2024
@author: louis
"""

import cv2
import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont, ImageColor
import os
import csv
import matplotlib.pyplot as plt
import string as str
def calculate_text_size(text, font):
    # calculate text size based on font properties
    ascent, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = ascent + descent
    return text_width, text_height

def get_brightness(color):
    # Calculate brightness of a color (grayscale value) for the text
    r, g, b = ImageColor.getrgb(color)
    return (r * 299 + g * 587 + b * 114) / 1000 


def visualize_image(filename, csv_filename):
        # Open image
        image_path = filename
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Read bounding box information from CSV file
        if os.path.getsize(csv_filename) > 0:
            with open(csv_filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                #next(csvreader)  # Skip header row
                for row in csvreader:
                    if row:
                        xmin, ymin, xmax, ymax = map(int, row[0:4])
                        class_name = row[4]
        
                        # Define colors for different classes
                        class_colors = {
                            'danger': 'yellow',
                            'interdiction': 'purple',
                            'obligation': 'blue',
                            'stop': 'magenta',
                            'ceder': 'cyan',
                            'frouge': 'red',
                            'forange': 'orange',
                            'fvert': 'green'
                        }
        
                         # Define brightness threshold for determining text color
                        brightness_threshold = 150  
        
                        # Get bounding box color
                        box_color = class_colors.get(class_name, 'white') #white is the de
        
                        # Determine text color based on brightness of box color
                        text_color = 'black' if get_brightness(box_color) > brightness_threshold else 'white'
        
                        # Draw bounding box
                        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=box_color)
        
                        # Define font and size
                        font_size = 30 # Adjust the font size here
                        font = ImageFont.truetype("arial.ttf", font_size)
        
                        # Get text size
                        text_width, text_height = calculate_text_size(class_name, font)
        
                        # Draw filled rectangle as background for class name
                        draw.rectangle([(xmin, ymin - text_height), (xmin + text_width, ymin)], fill=box_color)
        
                        # Draw class name text on top of the filled rectangle
                        draw.text((xmin, ymin - text_height), class_name, fill=text_color, font=font)
        return img

def preprocessing_data():
    label_to_int = {
    'danger': 1,
    'interdiction': 2,
    'obligation': 3,
    'stop': 4,
    'ceder': 5,
    'frouge': 6,
    'forange': 7,
    'fvert': 8,
    'ff':9,
}
    file_index = ["{:04d}".format(i) for i in range(1, 880)]
    imgs_names = []
    imgs_bb = []
    classes_indices=[[] for _ in range(11)]
    for i, file_name in enumerate(file_index):
        try:
            with open(f'train/labels/{file_name}.csv', 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                bounding_boxes = []
                for row in csvreader:
                    if row :   
                        xmin, ymin, xmax, ymax = map(int, row[0:4])
                        class_name = row[4].lower()
                        bounding_boxes.append([xmin, ymin, xmax, ymax, class_name])
                        classes_indices[label_to_int[class_name]].append(i)
                imgs_bb.append((bounding_boxes))
                imgs_names.append(f"train/images/{file_name}.jpg")
        except FileNotFoundError:
             classes_indices[10].append(i)

    return imgs_names, imgs_bb, classes_indices

# img = visualize_image("train/images/0001.jpg","train/labels/0001.csv")
# plt.figure()
# plt.imshow(img)
