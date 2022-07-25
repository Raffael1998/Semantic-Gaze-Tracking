from math import *
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import csv
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
csv.field_size_limit(9999999)

#Constants :
length_image = 1920
width_image = 1200

phi = 1.5 # pan angle of the camera around the y-axis (in degrees)
psi = 0 # tilt angle of the camera around the x-axis (in degrees)
theta = -1.15 # angle of the camera around the z-axis (in degrees)

transl_IBEO_cam = [1.87, 0.25, 1.1]
f = 0.012 # focal of the camera in meters
size_pixel = 5.86 * 10**(-6) # in meters

transl_GT_CAM = [-0.185, 0.224, 0.17] #l'origine du gaze tracker est centre avec l'axe x de IBEO : au centre de la voiture : -31cm par rapport aux mesures

height_IBEO = height_IBEO = 0.724/2


def ref_change_IBEOtoCAM(xyz_IBEO):

    def rot_x(psi):
        psi_ = radians(psi) # we need to convert psi to radians before using trigonometric functions
        return np.array([[1, 0, 0], [0, cos(psi_), -sin(psi_)], [0, sin(psi_), cos(psi_)]])
    def rot_y(phi):
        phi_ = radians(phi)
        return np.array([[cos(phi_), 0, sin(phi_)], [0, 1, 0], [-sin(phi_), 0, cos(phi_)]])
    def rot_z(theta):
        theta_ = radians(theta)
        return np.array([[cos(theta_), -sin(theta_), 0], [sin(theta_), cos(theta_), 0], [0, 0, 1]])

    xyz_IBEO = np.array(xyz_IBEO)
    t_IBEO_cam = np.array(transl_IBEO_cam)
    X_ = - t_IBEO_cam + xyz_IBEO
    X_cam = np.dot(rot_z(theta), np.dot(rot_y(phi), np.dot(rot_x(psi), X_)))
    return list(X_cam)

def ref_change_CAMtoSENSOR(xyz_cam):
    """
    Returns the (y_image, z_image) coordinates in the sensor for an object of coordinates (x, y, z) in the camera
    reference frame.
    """
    x_cam, y_cam, z_cam = xyz_cam
    y_sen = y_cam * f / x_cam
    z_sen = z_cam * f / x_cam
    return x_cam, -y_sen, -z_sen

def SENSORtoPIXEL(x_cam, y_sen, z_sen):
    """
    Returns the (y_pixel, z_pixel) coordinates in the image for an object of coordinates (y_im, z_im) in the sensor.
    """
    y_pix = int(y_sen / size_pixel) + length_image/2
    z_pix = int(z_sen / size_pixel) + width_image/2
    return x_cam, y_pix, z_pix

def projection_on_image(xyz_IBEO):
    """
    renvoit les coordonnes du pixel correspondant
    """
    if is_in_image(xyz_IBEO) :
        x_CAM, y_CAM, z_CAM = ref_change_IBEOtoCAM(xyz_IBEO)
        x_cam, y_im, z_im = ref_change_CAMtoSENSOR([x_CAM, y_CAM, z_CAM])
        x_cam, y_pix, z_pix =  SENSORtoPIXEL(x_cam, y_im, z_im)
        return [y_pix, z_pix]


def change_ref_GT_to_IBEO(X_GT):
  X_GT_fine = [-X_GT[2], -X_GT[0], X_GT[1]]
  X_GT_fine = np.array(X_GT_fine)
  ret = X_GT_fine - transl_GT_CAM + transl_IBEO_cam
  return ret


def is_in_image(xyzIBEO):
    if xyzIBEO == None:
        return None
    xyzCAM = ref_change_IBEOtoCAM(xyzIBEO)
    x_CAM = xyzCAM[0]
    y_sen, z_sen = ref_change_CAMtoSENSOR(xyzCAM)[1:]
    y_pix, z_pix = SENSORtoPIXEL(x_CAM, y_sen, z_sen)[1:]
    if x_CAM > 0:
        if 0 <= y_pix <= length_image-1 :
            if 0 <= z_pix <= width_image-1 :
                return True