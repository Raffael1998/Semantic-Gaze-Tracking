# Importing packages
from math import *
import numpy as np
from PIL import Image, ImageDraw
from csv import reader
import csv
import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
from sympy import li
csv.field_size_limit(9999999)
from shapely.geometry import Point, Polygon

from transforms import *

def import_files(files_list):
    ret = []
    for name in files_list:
        tempStr = []
        tempFloat = []
        with open(name, 'r') as read_obj:
            csv_reader = reader(read_obj)
            rows = list(csv_reader)

        for i in range(len(rows)):
            tempStr.append(list(rows[i][0].split(";")))
            line = []
            for j in range(len(tempStr[i])):
                line.append(float(tempStr[i][j]))
            tempFloat.append(line)
        ret.append(tempFloat)
    return ret

def find_closest_row(time, table):

    def get_closest_value(arr, target):
        n = len(arr)
        left = 0
        right = n - 1
        mid = 0

        # edge case - last or above all
        if target >= arr[n - 1]:
            return arr[n - 1]
        # edge case - first or below all
        if target <= arr[0]:
            return arr[0]
        # BSearch solution: Time & Space: Log(N)

        while left < right:
            mid = (left + right) // 2  # find the mid
            if target < arr[mid]:
                right = mid
            elif target > arr[mid]:
                left = mid + 1
            else:
                return arr[mid]

        if target < arr[mid]:
            return find_closest(arr[mid - 1], arr[mid], target)
        else:
            return find_closest(arr[mid], arr[mid + 1], target)


    def find_closest(val1, val2, target):
        return val2 if target - val1 >= val2 - target else val1

    time_list_GazeTracker = [float(table[i][0]) for i in range(len(table))]
    closest_time = get_closest_value(time_list_GazeTracker, time)
    return time_list_GazeTracker.index(closest_time)

def find_frame(time, video_file): #time in micro second #limit = 93217921
    vidcap = cv2.VideoCapture(video_file)
    if not vidcap.isOpened():
        exit(0)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)

    micro_seconds_per_frame = 1000000/fps
    frame_seq = (time // micro_seconds_per_frame) 
    vidcap.set(1,frame_seq)
    ret, frame = vidcap.read()
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vidcap.release()
    return pil_image

def find_frame_segm(time, images_path_list):
    length_video_micros = 93250000
    num_frame = int(len(images_path_list)*time/length_video_micros)
    frame = Image.open(images_path_list[num_frame])
    return frame

################ DRAWING // PLOTTING FUNCTIONS

def display_elipse_on_image(pixel, image, color, size):
    y_pix, z_pix = pixel
    draw = ImageDraw.Draw(image, 'RGBA')
    r = size
    leftUpPoint = (y_pix-r, z_pix-r)
    rightDownPoint = (y_pix+r, z_pix+r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=color)
    return image

def display_pixel_on_image(pixel, image, color, opacity):
    draw = ImageDraw.Draw(image, 'RGBA')
    colorRGB = hexa_color_to_rgb(color)
    colorRGBAlpha = colorRGB + [opacity]
    draw.point(pixel, fill = tuple(colorRGBAlpha))

def draw_box(image, box, color, size = 5):
    """
    funtion to draw a perspective box on a given image
    image (image object): a PIL image to draw on
    box (box object): a BoundingBox object that will be drawn
    color (str): a color (str) in hexadecimal
    size (int): the size of the dots that will make the bounding box

    Output (image object): PIL image
    """
    box_lines = box.bresenham_lines_bounding_box()
    for line in box_lines:
        for point in line:
            xyzIBEO = point
            if is_in_image(xyzIBEO):
                x_CAM, y_CAM, z_CAM = ref_change_IBEOtoCAM(xyzIBEO)
                x_cam, y_im, z_im = ref_change_CAMtoSENSOR([x_CAM, y_CAM, z_CAM])
                x_cam, y_pix, z_pix =  SENSORtoPIXEL(x_cam, y_im, z_im)
                #display_pixel_on_image([y_pix, z_pix], frame, '#3cb44b')
                display_elipse_on_image([y_pix, z_pix], image, color, size)

def draw_gaze(image, gaze_ray_list, color, size = 8):
    for gaze_ray in gaze_ray_list:
        if gaze_ray.hit_point != None:
            if is_in_image(gaze_ray.hit_point):
                y_pix, z_pix = projection_on_image(gaze_ray.hit_point)
                gaze_ray.corresponding_pixel_y, gaze_ray.corresponding_pixel_z = y_pix, z_pix
                display_point_on_image([y_pix, z_pix], image, color, size)

def draw_gaze_with_no_hitpoints(image, gaze_ray_list, colorhit, colotNohit, size = 8):
    for gaze_ray in gaze_ray_list:
        if gaze_ray.hit_point != None:
            if is_in_image(gaze_ray.hit_point):
                y_pix, z_pix = projection_on_image(gaze_ray.hit_point)
                gaze_ray.corresponding_pixel_y, gaze_ray.corresponding_pixel_z = y_pix, z_pix
                display_point_on_image([y_pix, z_pix], image, colorhit, size)
        else:
            x0, y0,z0 = gaze_ray.head_pos_IBEO
            dx, dy, dz = gaze_ray.gaze_vec_IBEO
            lam = 100 #default value of gaze ray length is 100m
            hit_point = [x0 + lam*dx, y0 + lam*dy, z0 + lam*dz]
            gaze_ray.hit_point = hit_point #we create an artificial gaze point
            if is_in_image(gaze_ray.hit_point):
                y_pix, z_pix = projection_on_image(gaze_ray.hit_point)
                gaze_ray.corresponding_pixel_y, gaze_ray.corresponding_pixel_z = y_pix, z_pix
                display_point_on_image([y_pix, z_pix], image, colotNohit, size)

def draw_gaze_unique(image, gaze_ray, color, size = 8):
    if gaze_ray.hit_point != None:
        if is_in_image(gaze_ray.hit_point):
            y_pix, z_pix = projection_on_image(gaze_ray.hit_point)
            display_point_on_image([y_pix, z_pix], image, color, size)

def display_point_on_image(pixel, image, color, size = 20): 
    """
    funtion to draw a dot at a given pixel on a given image
    pixel (2-tuple): the coordinates of the point
    image (image object): a PIL image to draw on
    color (str): a color (str) in hexadecimal
    size (int): the size of the dot

    Output (image object): PIL image
    """
    y_pix, z_pix =  pixel
    draw = ImageDraw.Draw(image, 'RGBA')
    r = size
    leftUpPoint = (y_pix-r, z_pix-r)
    rightDownPoint = (y_pix+r, z_pix+r)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=color)
    return image

def plot_box_3D(axes, box, color):
    edges = box.edges_bounding_box()
    for SingleEdgeVertexPair in edges:

        Vertex1 = SingleEdgeVertexPair[0]
        Vertex2 = SingleEdgeVertexPair[1]

        EdgeXvals = [Vertex1[0], Vertex2[0] ]
        EdgeYvals = [Vertex1[1], Vertex2[1] ]
        EdgeZvals = [Vertex1[2], Vertex2[2] ]

        axes.plot(EdgeXvals, EdgeYvals, EdgeZvals, c=color, marker=None, linestyle = '-', linewidth = 0.9)

def plot_line_3D(axes, origin, vec_dir, length=50, color='#000000', linewidth = 2):
    x0, y0, z0 = origin

    axes.plot([x0, x0+vec_dir[0]*length], [y0, y0+vec_dir[1]*length], [z0, z0+vec_dir[2]*length], c=color, marker=None, linestyle = '-', linewidth = linewidth)

def is_in_shape(pix_y, pix_z, shape_point_list):
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2':"""
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    bool = True
    for i in range(len(shape_point_list)):
        y_0, z_0 = shape_point_list[i-1]
        y_1, z_1 = shape_point_list[i]
        u = np.array([pix_y-y_0, pix_z-z_0])
        v = np.array([y_1-y_0, z_1-z_0])
        sine = np.sin(angle_between(u, v))
        if sine < 0:
            bool = bool*False #if one of the test is negative : the bool is set to False and stays that way
    return bool

def is_in_shape2(pix_y, pix_z, shape_point_list):
    p1 = Point(pix_y, pix_z)
    poly = Polygon(shape_point_list)
    return poly.contains(p1)
        
def find_pixels_cone(gaze_ray_list):
    pixel_list = []
    pixely_list = []
    pixelz_list = []
    point_list = []
    for gaze_ray in gaze_ray_list[1:]:
        if (gaze_ray.corresponding_pixel_y != None) and (gaze_ray.corresponding_pixel_z != None):
            pixely_list.append(gaze_ray.corresponding_pixel_y)
            pixelz_list.append(gaze_ray.corresponding_pixel_z)
            point_list.append([gaze_ray.corresponding_pixel_y, gaze_ray.corresponding_pixel_z])
    if len(pixely_list) > 3  and len(pixelz_list) > 3: #on affiche la shape du regard que lorsqu'on a 4 points ou plus dans l'image
        pixely_min, pixely_max = int(min(pixely_list)), int(max(pixely_list))
        pixelz_min, pixelz_max = int(min(pixelz_list)), int(max(pixelz_list))
        for i in range(pixely_min, pixely_max+1):
            for j in range(pixelz_min, pixelz_max+1):
                if is_in_shape2(i, j, point_list):
                    if i < length_image:
                        if j < width_image:
                            pixel_list.append([i, j])

    return pixel_list


############ RANDOM UTILS
def longest_key(dict): #outputs the key that has the longest list as value in a dictionnary
    longest = max(len(item) for item in dict.values())
    max_key = list(dict.keys())[0]
    for key in list(dict.keys()):
        if len(dict[key]) > len(dict[max_key]):
             max_key = key
    return max_key

def hexa_color_to_rgb(hexa):
    return list(int(hexa.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))