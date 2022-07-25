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

def main():
    for line_num in range(len(objectBoxCenters))[200:]:

        #CREATE THE VALUES LISTS
        x_list_points = [float(objectBoxCenters[line_num][k]) for k in range(len(objectBoxCenters[line_num])) if k % 2 == 1]
        y_list_points = [float(objectBoxCenters[line_num][k]) for k in range(1, len(objectBoxCenters[line_num])) if k % 2 == 0]

        length_list = [float(objectBoxSizes[line_num][k]) for k in range(len(objectBoxCenters[line_num])) if k % 2 == 1]
        width_list = [float(objectBoxSizes[line_num][k]) for k in range(1, len(objectBoxCenters[line_num])) if k % 2 == 0]

        angle_list = [float(objectBoxOrientations[line_num][k]) for k in range(1, len(objectBoxOrientations[line_num]))]

        classes = [classifications[line_num][k] for k in range(1, len(classifications[line_num]))]

        velocity_x = [float(absoluteVelocities[line_num][k]) for k in range(len(absoluteVelocities[line_num])) if k % 2 == 1]
        velocity_y = [float(absoluteVelocities[line_num][k]) for k in range(1, len(absoluteVelocities[line_num])) if k % 2 == 0]
        absoluteVelocity = [sqrt((vx**2) + (vy**2)) for vx, vy in zip(velocity_x, velocity_y)]

        #CREATE THE BOX LIST
        box_list = []
        for i in range(len(x_list_points)):
            box = BoundingBox(x_center=x_list_points[i], y_center=y_list_points[i], z=-height_IBEO, length=length_list[i], width=width_list[i], angle=angle_list[i], classe=int(classes[i]), color=color_list[int(classes[i])], absoluteVelocity=absoluteVelocity[i])
            box_list.append(box)

        time = float(objectBoxCenters[line_num][0])
        video_file = r'C:\Users\grosr\OneDrive - Queensland University of Technology\Desktop\Raffael\CARRS-Q internship\Data\DataRaw\PC2RecVideoDMS_20220330_163312_FrontCam_imageOut.avi'
        frame = find_frame(time, video_file)


        #ADD CUSTOM BOXES
        road = BoundingBox(x_center=0, y_center=0, z=-height_IBEO, length=120, width=16, angle=0, classe=8, absoluteVelocity=10, height=0)
        box_list.insert(0, road)

        #invisible_wall = BoundingBox(x_center=1000, y_center= 0, z=-250, length=0.1, width=500, angle=0, classe=8, color= color_list[10], absoluteVelocity= 10)
        #invisible_wall.height = 500
        #box_list.insert(0, invisible_wall)

        #DEAL WITH GAZE
        gaze_ray_list = gaze_creator(9, gaze_list_float[find_closest_row(time, gaze_list_float)], angle_deg=3)
        gaze_ray = gaze_ray_list[0]
        head_pos = gaze_ray.head_pos_IBEO

        #CONFIGURE THE PLOT
        plt.figure(figsize=(10,10))
        axes = plt.subplot(111, projection='3d')
        axes.set_xlim([-10, 30])
        axes.set_ylim([-20, 20])
        axes.set_zlim([-5, 35])
        plt.xlabel('x')
        plt.ylabel('y')
        axes.view_init(elev=20, azim=200)

        #PLOT THE BOXES
        closest_box_seen = Looking_At(gaze_ray, box_list)
        for i in range(len(box_list)):
            box = box_list[i]
            if closest_box_seen == box :
                plot_box_3D(axes, box, '#3cb44b')
            else:
                plot_box_3D(axes, box, box.color)

        #PLOT ZOE
        plot_box_3D(axes, Zoe, Zoe.color)
        
        #PLOT GAZE LINE
        plot_line_3D(axes, head_pos, gaze_ray.gaze_vec_IBEO, length=50, color='#000000', linewidth = 0.5)
        axes.scatter(head_pos[0], head_pos[1], head_pos[2], c = '#911eb4', s = 4)

        #PLOT THE INTERSECTION POINT
        if gaze_ray.hit_point != None:
            axes.scatter(gaze_ray.hit_point[0], gaze_ray.hit_point[1], gaze_ray.hit_point[2], c = '#911eb4', s = 4)


        print('fig' + "%05d"%line_num + '.png')
        plt.savefig(r'test\fig%05d.png' %line_num)
        #plt.show()

if __name__ == "__main__":
    main()