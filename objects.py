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
csv.field_size_limit(9999999)

from transforms import *
from utils import *

color_list = ['#e6194B', '#dcbeff', '#4363d8', '#f58231', '#911eb4', '#f032e6', '#fabed4', '#469990', '#dcbeff', 
            '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

class BoundingBox: 
        def __init__(self,    x_center, y_center, z, length, width, angle, absoluteVelocity, classe, color=None, height=None):
                self.x_center = x_center
                self.y_center = y_center
                self.length = length
                self.width = width
                self.angle = angle
                self.classe = classe
                self.absoluteVelocity = absoluteVelocity
                self.z = z


                self.color = color_list[classe] if color is None else color

                self.first_intersect = False #attribute that tells if the box is the first intersected by the gaze
                self.hit_point = None
                self.distance_gaze = None


                self.height = compute_height(self.classe) if height is None else height


        def create_base_rectangle(self):

                x_1 = self.x_center - self.length/2
                x_2 = self.x_center + self.length/2
                y_1 = self.y_center - self.width/2
                y_2 = self.y_center + self.width/2

                angle_rd = np.radians(self.angle)
                rot_matrix = np.array([(np.cos(angle_rd), -np.sin(angle_rd)),
                                                                (np.sin(angle_rd), np.cos(angle_rd))])
                
                x_center_1 = x_1 - self.x_center
                y_center_1 = y_1 - self.y_center
                x_center_2 = x_2 - self.x_center
                y_center_2 = y_2 - self.y_center

                #coordinates of the 4 base points : pt1, 2, 3, 4
                pt1 = rot_matrix.dot(np.array([x_center_1, y_center_1])) + np.array([self.x_center, self.y_center])
                pt2 = rot_matrix.dot(np.array([x_center_2, y_center_1])) + np.array([self.x_center, self.y_center])
                pt3 = rot_matrix.dot(np.array([x_center_2, y_center_2])) + np.array([self.x_center, self.y_center])
                pt4 = rot_matrix.dot(np.array([x_center_1, y_center_2])) + np.array([self.x_center, self.y_center])

                vertices_list = []
                vertices_list.append(pt1)
                vertices_list.append(pt2)
                vertices_list.append(pt3)
                vertices_list.append(pt4)

                return vertices_list


        def points_bounding_box(self):
                pt1, pt2, pt3, pt4 = self.create_base_rectangle()
                pt1 = np.append(pt1, self.z) #transform to 3D array
                pt2 = np.append(pt2, self.z)
                pt3 = np.append(pt3, self.z)
                pt4 = np.append(pt4, self.z)
                height_vec = np.array([0, 0, self.height])
                #coordinates of the upper points
                pt5, pt6, pt7, pt8 = pt1 + height_vec, pt2 + height_vec, pt3 + height_vec, pt4 + height_vec
                return pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8


        def edges_bounding_box(self):
                pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8 = self.points_bounding_box()

                edges = []
                edges.append([pt1, pt2])
                edges.append([pt1, pt4])
                edges.append([pt1, pt5])
                edges.append([pt2, pt3])
                edges.append([pt3, pt4])
                edges.append([pt2, pt6])
                edges.append([pt3, pt7])
                edges.append([pt4, pt8])
                edges.append([pt5, pt6])
                edges.append([pt5, pt8])
                edges.append([pt6, pt7])
                edges.append([pt7, pt8])
                return edges


        def bresenham_lines_bounding_box(self):

                def Bresenham3D_inDM(pt1, pt2): # input in meter
                        x1, y1, z1 = pt1
                        x2, y2, z2 = pt2
                        x1 = int(x1*10)
                        y1 = int(y1*10)
                        z1 = int(z1*10)
                        x2 = int(x2*10)
                        y2 = int(y2*10)
                        z2 = int(z2*10)

                        ListOfPoints = []
                        ListOfPoints.append((x1, y1, z1))
                        dx = abs(x2 - x1)
                        dy = abs(y2 - y1)
                        dz = abs(z2 - z1)
                        if (x2 > x1):
                                xs = 1
                        else:
                                xs = -1
                        if (y2 > y1):
                                ys = 1
                        else:
                                ys = -1
                        if (z2 > z1):
                                zs = 1
                        else:
                                zs = -1

                        # Driving axis is X-axis"
                        if (dx >= dy and dx >= dz):		
                                p1 = 2 * dy - dx
                                p2 = 2 * dz - dx
                                while (x1 != x2):
                                        x1 += xs
                                        if (p1 >= 0):
                                                y1 += ys
                                                p1 -= 2 * dx
                                        if (p2 >= 0):
                                                z1 += zs
                                                p2 -= 2 * dx
                                        p1 += 2 * dy
                                        p2 += 2 * dz
                                        ListOfPoints.append((x1, y1, z1))

                        # Driving axis is Y-axis"
                        elif (dy >= dx and dy >= dz):	
                                p1 = 2 * dx - dy
                                p2 = 2 * dz - dy
                                while (y1 != y2):
                                        y1 += ys
                                        if (p1 >= 0):
                                                x1 += xs
                                                p1 -= 2 * dy
                                        if (p2 >= 0):
                                                z1 += zs
                                                p2 -= 2 * dy
                                        p1 += 2 * dx
                                        p2 += 2 * dz
                                        ListOfPoints.append((x1, y1, z1))

                        # Driving axis is Z-axis"
                        else:		
                                p1 = 2 * dy - dz
                                p2 = 2 * dx - dz
                                while (z1 != z2):
                                        z1 += zs
                                        if (p1 >= 0):
                                                y1 += ys
                                                p1 -= 2 * dz
                                        if (p2 >= 0):
                                                x1 += xs
                                                p2 -= 2 * dz
                                        p1 += 2 * dy
                                        p2 += 2 * dx
                                        ListOfPoints.append((x1, y1, z1))
                        
                        ret =[]
                        for L in ListOfPoints:
                                temp = []
                                for val in L:
                                        val = val/10
                                        temp.append(val)
                                ret.append(temp)
                        return ret

                pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8 = self.points_bounding_box()
                lines = []
                lines.append(Bresenham3D_inDM(pt1, pt2))
                lines.append(Bresenham3D_inDM(pt1, pt4))
                lines.append(Bresenham3D_inDM(pt1, pt5))
                lines.append(Bresenham3D_inDM(pt2, pt3))
                lines.append(Bresenham3D_inDM(pt3, pt4))
                lines.append(Bresenham3D_inDM(pt2, pt6))
                lines.append(Bresenham3D_inDM(pt3, pt7))
                lines.append(Bresenham3D_inDM(pt4, pt8))
                lines.append(Bresenham3D_inDM(pt5, pt6))
                lines.append(Bresenham3D_inDM(pt5, pt8))
                lines.append(Bresenham3D_inDM(pt6, pt7))
                lines.append(Bresenham3D_inDM(pt7, pt8))
                return lines

class Gaze:
    def __init__(self, head_pos_IBEO, gaze_vec, ray_id):
        self.ray_id = ray_id

        self.head_pos_IBEO = head_pos_IBEO
        self.gaze_vec_IBEO = gaze_vec

        self.hit_point = None
        self.box_hit = None

        self.corresponding_pixel_y = None
        self.corresponding_pixel_z = None

def gaze_creator(number_of_rays, gaze_float, angle_deg = 3): #number_of_rays = 1 or 9
        gaze_ray_list = []

        head_pos_GT = gaze_float[1:4]
        gaze_vec_GT = gaze_float[4:7]
        head_pos_IBEO = change_ref_GT_to_IBEO(head_pos_GT)
        gaze_vec_IBEO = [-gaze_vec_GT[2], -gaze_vec_GT[0], gaze_vec_GT[1]]

        dx, dy, dz = gaze_vec_IBEO


        gaze_ray_list.append(Gaze(head_pos_IBEO, gaze_vec_IBEO, ray_id=0)) #we add the first ray (the main one in the middle) to the list


        if (dx, dy) != (0, 0) and (dx, dz) != (0, 0) :
            theta = np.radians(angle_deg)
            r = np.tan(theta)
            vec = np.array(gaze_vec_IBEO)
            vz = np.array([(-r*dz)/sqrt(dx**2 + dz**2), 0, (r*dx)/sqrt(dx**2 + dz**2)])
            vy = np.array([(-r*dy)/sqrt(dx**2 + dy**2), (r*dx)/sqrt(dx**2 + dy**2), 0])
            A = sqrt(2)/2

            v0 = vec
            v1 = vec + vz
            v2 = vec + A*(vz - vy)
            v3 = vec - vy
            v4 = vec - A*(vz + vy)
            v5 = vec -vz
            v6 = vec + A*(vy - vz)
            v7 = vec + vy
            v8 = vec + A*(vy + vz)
            vec_list = [list(v0), list(v1), list(v2), list(v3), list(v4), list(v5), list(v6), list(v7), list(v8)]

            for i in range(len(vec_list))[1:number_of_rays]:
                gaze_ray_list.append(Gaze(head_pos_IBEO, vec_list[i], ray_id=i)) #we add the other rays to the list
        
        return gaze_ray_list
            

def do_intersect_box(bounding_box, head_pos, gaze_vec):
    lambda_list = []
    index_list = [] #list to know wich side it intersected with
    hit_points = []
    x0, y0, z0 = head_pos
    dx, dy, dz = gaze_vec
    pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8 = bounding_box.points_bounding_box()
    if sqrt((dx**2) + (dy**2)) != 0: #si le vecteur du gaze a ete capte
        #disjonction de 3 cas : selon la direction de la face du cub regarde
        for i in range(3):
            for alpha in [0,1]: #disjonction de la face avant ou arriere
                if i == 0: #face normale a z
                    xA, yA, zA = pt1 #pt1 base_point
                    xB, yB, zB = pt2
                    xD, yD, zD = pt4
                    xE, yE, zE = pt5
                    Y = np.array([[x0 - xA - alpha*(xE - xA)], 
                                                [y0 - yA - alpha*(yE - yA)],
                                                [z0 - zA - alpha*(zE - zA)]])

                    A = np.array([[-dx, xB - xA, xD - xA], 
                                                [-dy, yB - yA, yD - yA],
                                                [-dz, zB - zA, zD - zA]])
                    
                    if np.linalg.cond(A) < 1/sys.float_info.epsilon: #matix invertible : gaze not parallel with plane
                        X = np.linalg.inv(A).dot(Y)
                        
                        lam = X[0][0]
                        mu1 = X[1][0]
                        mu2 = X[2][0]
                        if (lam > 0) and (0 <= mu1 <= 1) and (0 <= mu2 <= 1):
                            lambda_list.append(lam)
                            index_list.append([i, alpha])
                            hit_points.append([x0 + lam*dx, y0 + lam*dy, z0 + lam*dz])

                if i == 1: #face normale a x
                    xA, yA, zA = pt1 #pt1 base_point
                    xD, yD, zD = pt4
                    xE, yE, zE = pt5

                    xB, yB, zB = pt2
                    Y = np.array([[x0 - xA - alpha*(xB - xA)], 
                                                [y0 - yA - alpha*(yB - yA)],
                                                [z0 - zA - alpha*(zB - zA)]])

                    A = np.array([[-dx, xD - xA, xE - xA], 
                                                [-dy, yD - yA, yE - yA],
                                                [-dz, zD - zA, zE - zA]])
                    
                    if np.linalg.cond(A) < 1/sys.float_info.epsilon: #matix invertible : gaze not parallel with plane
                        X = np.linalg.inv(A).dot(Y)
                        
                        lam = X[0][0]
                        mu1 = X[1][0]
                        mu2 = X[2][0]
                        if (lam > 0) and (0 <= mu1 <= 1) and (0 <= mu2 <= 1):
                            lambda_list.append(lam)
                            index_list.append([i, alpha])
                            hit_points.append([x0 + lam*dx, y0 + lam*dy, z0 + lam*dz])

                if i == 2: #face normale a y
                    xA, yA, zA = pt1 #pt1 base_point
                    xB, yB, zB = pt2
                    xE, yE, zE = pt5

                    xD, yD, zD = pt4
                    Y = np.array([[x0 - xA - alpha*(xD - xA)], 
                                                [y0 - yA - alpha*(yD - yA)],
                                                [z0 - zA - alpha*(zD - zA)]])

                    A = np.array([[-dx, xB - xA, xE - xA], 
                                                [-dy, yB - yA, yE - yA],
                                                [-dz, zB - zA, zE - zA]])
                    
                    if np.linalg.cond(A) < 1/sys.float_info.epsilon: #matix invertible : gaze not parallel with plane
                        X = np.linalg.inv(A).dot(Y)
                        
                        lam = X[0][0]
                        mu1 = X[1][0]
                        mu2 = X[2][0]
                        
                        if (lam > 0) and (0 <= mu1 <= 1) and (0 <= mu2 <= 1):
                            lambda_list.append(lam)
                            index_list.append([i, alpha])
                            hit_points.append([x0 + lam*dx, y0 + lam*dy, z0 + lam*dz])

        if lambda_list == []: return False
        else:
                lambda_list, index_list, hit_points = (list(t) for t in zip(*sorted(zip(lambda_list, index_list, hit_points))))
                return lambda_list[0], index_list[0], hit_points[0]

def Looking_At(gaze_ray, box_list): #they must be in the IBEO ref

        closest_box = None
        head_pos = gaze_ray.head_pos_IBEO
        gaze_vec = gaze_ray.gaze_vec_IBEO
        for i in range(len(box_list)):
            box = box_list[i]
            if do_intersect_box(box, head_pos, gaze_vec):
                box.distance_gaze = do_intersect_box(box, head_pos, gaze_vec)[0]
                box.hit_point = do_intersect_box(box, head_pos, gaze_vec)[2]
                if closest_box == None:
                    closest_box = box
                    gaze_ray.hit_point = do_intersect_box(box, head_pos, gaze_vec)[2] #we put the box hit and the hitpoint in the hitpointdict of the gaze
                    gaze_ray.box_hit = box
                elif box.distance_gaze < closest_box.distance_gaze:
                    #we update the closest box and we remove the closest box status from the previous one
                    closest_box = box
                    gaze_ray.hit_point = do_intersect_box(box, head_pos, gaze_vec)[2] #we put the box hit and the hitpoint in the hitpointdict of the gaze
                    gaze_ray.box_hit = box
        return closest_box


def Looking_At_SeveralGazeLines(gaze_ray_list, box_list):
    dic_object = {}
    for gaze_ray in gaze_ray_list:
        object = Looking_At(gaze_ray, box_list) #this updates the gaze by adding the hitpoints to the hitpointdict
        if object != None:
            dic_object.setdefault(object,[]).append(gaze_ray) #we only take into consideration the object (we remove None from the keys)
    if dic_object == {}:
        return None
    else:
        max_gaze_object = longest_key(dic_object) #outputs the object with the maximum amount of rays on it
        return max_gaze_object


def Looking_At_SeveralGazeLines_all_boxes_seen(gaze_ray_list, box_list): #outputs the entire ditionnary
    dic_object = {}
    for gaze_ray in gaze_ray_list:
        object = Looking_At(gaze_ray, box_list) #this updates the gaze by adding the hitpoints to the hitpointdict
        if object != None:
            dic_object.setdefault(object,[]).append(gaze_ray) #we only take into consideration the object (we remove None from the keys)

    return dic_object



def compute_height(class_num):
    if class_num == 5: #car
        return 1.562
    elif class_num == 2:
        return 5
    elif class_num == 1:
        return 0.5
    else:
        return 1.5


#create the object Zoe2
dist_IBEO_arriere = 0.661
l_Zoe = 4.087
dist_front = l_Zoe-dist_IBEO_arriere
w_Zoe = 1.94
h_Zoe = 1.562
center_car_x = (l_Zoe/2)-dist_IBEO_arriere

Zoe = BoundingBox(x_center=center_car_x, y_center=0, z=-height_IBEO, length=l_Zoe, width=w_Zoe, angle=0, absoluteVelocity=5, classe=5, color='#e6194B', height=h_Zoe)