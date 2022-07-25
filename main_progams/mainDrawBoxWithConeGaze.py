from objects import *
from transforms import *
from utils import *

objectBoxCenters_csv = r"C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\IbeoData\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_objectBoxCenters.csv"
classifications_csv = r"C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\IbeoData\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_classifications.csv"
objectBoxSizes_csv = r"C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\IbeoData\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_objectBoxSizes.csv"
objectBoxOrientations_csv = r"C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\IbeoData\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_objectBoxOrientations.csv"
absoluteVelocities_csv = r"C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\IbeoData\RecFile_1_20220601_164333_IbeoObjectsSplitter_1_absoluteVelocities.csv"
gaze_list_csv = r"C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\DataRaw\PC2RecVideoDMS_20220330_163312_DMS_ORIG_DEST_o_data (Driver's Gaze).csv"

[objectBoxCenters, classifications, objectBoxSizes, 
    objectBoxOrientations, absoluteVelocities, gaze_list_float] = import_files([objectBoxCenters_csv, classifications_csv, objectBoxSizes_csv, 
                                                                                objectBoxOrientations_csv, absoluteVelocities_csv, gaze_list_csv])


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
        video_file = r'C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\DataRaw\PC2RecVideoDMS_20220330_163312_FrontCam_imageOut.avi'
        frame = find_frame(time-objectBoxCenters[0][0], video_file) #time adjust


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
                draw_box(frame, box, '#3cb44b', size = 5)
            else:
                draw_box(frame, box, box.color, size = 5)


        draw_gaze(frame, gaze_ray_list, '#aaffc3', size = 8)


        frame.save(r'test\image%05d.png' %line_num)
        print('image' + "%05d"%line_num + '.png')

if __name__ == "__main__":
    main()