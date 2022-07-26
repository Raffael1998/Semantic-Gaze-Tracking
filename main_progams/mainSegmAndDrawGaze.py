from PIL import Image, ImageFont, ImageDraw

from objects import *
from transforms import *
from utils import *
from segmentation import *

objectBoxCenters_csv = r".\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_objectBoxCenters.csv"
classifications_csv = r".\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_classifications.csv"
objectBoxSizes_csv = r".\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_objectBoxSizes.csv"
objectBoxOrientations_csv = r".\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_objectBoxOrientations.csv"
absoluteVelocities_csv = r".\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_absoluteVelocities.csv"
gaze_list_csv = r".\PC2RecVideoDMS_20220330_163312_DMS_ORIG_DEST_o_data (Driver's Gaze).csv"

[objectBoxCenters, classifications, objectBoxSizes, 
    objectBoxOrientations, absoluteVelocities, gaze_list_float] = import_files([objectBoxCenters_csv, classifications_csv, objectBoxSizes_csv, 
                                                                                objectBoxOrientations_csv, absoluteVelocities_csv, gaze_list_csv])

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


ONLY_MOVING_BOXES = False

def main():
    for line_num in range(len(objectBoxCenters))[600:]:
        x_list_points = [float(objectBoxCenters[line_num][k]) for k in range(len(objectBoxCenters[line_num])) if k % 2 == 1]
        y_list_points = [float(objectBoxCenters[line_num][k]) for k in range(1, len(objectBoxCenters[line_num])) if k % 2 == 0]

        length_list = [float(objectBoxSizes[line_num][k]) for k in range(len(objectBoxCenters[line_num])) if k % 2 == 1]
        width_list = [float(objectBoxSizes[line_num][k]) for k in range(1, len(objectBoxCenters[line_num])) if k % 2 == 0]

        angle_list = [float(objectBoxOrientations[line_num][k]) for k in range(1, len(objectBoxOrientations[line_num]))]

        velocity_x = [float(absoluteVelocities[line_num][k]) for k in range(len(absoluteVelocities[line_num])) if k % 2 == 1]
        velocity_y = [float(absoluteVelocities[line_num][k]) for k in range(1, len(absoluteVelocities[line_num])) if k % 2 == 0]
        absoluteVelocity = [sqrt((vx**2) + (vy**2)) for vx, vy in zip(velocity_x, velocity_y)]

        classes = [classifications[line_num][k] for k in range(1, len(classifications[line_num]))]

        box_list = []

        for i in range(len(x_list_points)):
            box = BoundingBox(x_center=x_list_points[i], y_center=y_list_points[i], z=-height_IBEO, length=length_list[i], width=width_list[i], angle=angle_list[i], classe=int(classes[i]), color=color_list[int(classes[i])], absoluteVelocity=absoluteVelocity[i])
            if box.absoluteVelocity >= 0.5 and ONLY_MOVING_BOXES: #we only consider the moving objects
                box_list.append(box)
            else : box_list.append(box)

        time = float(objectBoxCenters[line_num][0])


        video_file = r'.\PC2RecVideoDMS_20220330_163312_FrontCam_imageOut.avi'
        frame = find_frame(time, video_file)
        pred_segmented = create_pred(frame)
        image_segmented = color_image(pred_segmented)



        #ADD CUSTOM BOXES
        road = BoundingBox(x_center=0, y_center=0, z=-height_IBEO, length=80, width=16, angle=0, classe=8, absoluteVelocity=10, height=0)
        box_list.insert(0, road)
        #invisible_wall = BoundingBox(x_center=1000, y_center= 0, z=-250, length=0.1, width=500, angle=0, classe=8, color= color_list[10], absoluteVelocity= 10)
        #invisible_wall.height = 500
        #box_list.insert(0, invisible_wall)

        #DEAL WITH GAZE
        gaze_ray_list = gaze_creator(9, gaze_list_float[find_closest_row(time, gaze_list_float)], angle_deg=3)

        #PLOT THE BOXES AND INTERSECTION POINTS
        box_seen_dict = Looking_At_SeveralGazeLines_all_boxes_seen(gaze_ray_list, box_list)
        box_seen_list = list(box_seen_dict.keys())
        for i in range(len(box_list)):
            box = box_list[i]
            if box in box_seen_list :
                draw_box(image_segmented, box, '#3cb44b', size = 5)
            else:
                draw_box(image_segmented, box, box.color, size = 5)


        draw_gaze_with_no_hitpoints(image_segmented, gaze_ray_list, '#aaffc3', '#e6194B', size = 5)
    



        gaze_pixel_list = find_pixels_cone(gaze_ray_list)
        for pixel in gaze_pixel_list:
            display_pixel_on_image(pixel, image_segmented, '#ffffff', opacity = 100)


        ###########################################################################################################################

        #function to create the text displayed on the image
        def compute_MostRepresentedClasses(gaze_pixel_list, pred):
            class_names = ['road' ,'sidewalk' ,'building' ,'wall' ,'fence' ,'pole' ,'traffic light' ,'traffic sign' ,'vegetation' ,'terrain' ,
                        'sky' ,'person' ,'rider' ,'car' ,'truck' , 'bus' , 'train', 'motorcycle' , 'bicycle' ]

            #we create a dictionnary that contains the number of occurences of the classes inside the shape of the gaze
            num_pixel_total = len(gaze_pixel_list)
            class_dict = {}
            for pixel in gaze_pixel_list:
                pix_y, pix_z = pixel
                class_pixel = pred[pix_z, pix_y]
                if class_names[class_pixel] not in list(class_dict.keys()):
                    class_dict[class_names[class_pixel]] = 1
                else : 
                    class_dict[class_names[class_pixel]] +=1
            #we sort the dictionnary by occurences
            class_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1], reverse=True)}

            Looking_At_string = 'Looking at : '
            if 'road' in list(class_dict.keys()):
                Looking_At_string += 'road ; '
            if 'car' in list(class_dict.keys()):
                Looking_At_string += 'car ; '
            if 'traffic sign' in list(class_dict.keys()):
                Looking_At_string += 'traffic sign ; '

            text = ''
            for key in class_dict:
                text += key + ' : ' + str(int(class_dict[key] * 100/num_pixel_total)) + '%' + '\n'

            return Looking_At_string, text

        Looking_At_string, Percentages = compute_MostRepresentedClasses(gaze_pixel_list, pred_segmented)
            
        font2 = ImageFont.truetype('arial.ttf', 20)
        draw = ImageDraw.Draw(image_segmented, 'RGBA')
        draw.text((1770, 10), Percentages, font = font2)

        font1 = ImageFont.truetype('arial.ttf', 80)
        draw.text((50, 50), Looking_At_string, font = font1)

        image_segmented.save(r'.\output\image%05d.png' %line_num)
        print('image' + "%05d"%line_num + '.png')

if __name__ == "__main__":
    main()