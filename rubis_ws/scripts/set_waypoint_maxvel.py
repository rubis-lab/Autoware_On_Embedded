import numpy as np
import matplotlib.pyplot as plt
import csv
import yaml
import os

straight_area = []
curve_area = []

# buffer : middle area between curve and straight area

########### TODO ###########

########## FMTC RED Course Setting ##########

# point_csv_path = os.environ["HOME"] + "/autoware.ai/autoware_files/vector_map/220203_fmtc_red/point.csv"
# yaml_path = os.environ["HOME"] + "/rubis_ws/src/rubis_autorunner/cfg/ionic_autorunner/ionic_FMTC_red_course_vel.yaml"

# use_algorithm = True

# curve_velocity = 2
# buffer_velocity = 3
# straight_velocity = 5

# straight_area.append([1, 215])
# straight_area.append([270, 288])
# straight_area.append([330, 445])
# straight_area.append([600, 637])

# curve_area.append([236, 260])
# curve_area.append([295, 317])
# curve_area.append([465, 584])

#############################################

########## CubeTown Setting ##########

# point_csv_path = os.environ["HOME"] + "/autoware.ai/autoware_files/vector_map/cubetown_circle/point.csv"
# yaml_path = os.environ["HOME"] + "/rubis_ws/src/rubis_autorunner/cfg/cubetown_autorunner/cubetown_vel.yaml"

# use_algorithm = True

# curve_velocity = 2
# buffer_velocity = 4
# straight_velocity = 7

# straight_area.append([1, 20])
# straight_area.append([88, 130])
# straight_area.append([195, 218])
# straight_area.append([250, 318])

# curve_area.append([40, 75])
# curve_area.append([145, 180])
# curve_area.append([230, 260])
# curve_area.append([330, 362])

#############################################

########## 138 Ground Single Curve Setting ##########

# point_csv_path = os.environ["HOME"] + "/autoware.ai/autoware_files/vector_map/220118_138ground/single_curve/point.csv"
# yaml_path = os.environ["HOME"] + "/rubis_ws/src/rubis_autorunner/cfg/ionic_autorunner/138ground_single_curve_vel.yaml"

# use_algorithm = True

# curve_velocity = 2
# buffer_velocity = 4
# straight_velocity = 7

# straight_area.append([1, 10])

# curve_area.append([15, 40])

#############################################

########## 138 Ground Double Curve Setting ##########

point_csv_path = os.environ["HOME"] + "/autoware.ai/autoware_files/vector_map/220118_138ground/double_curve/point.csv"
yaml_path = os.environ["HOME"] + "/rubis_ws/src/rubis_autorunner/cfg/ionic_autorunner/138ground_double_curve_vel.yaml"

use_algorithm = True

curve_velocity = 2
buffer_velocity = 4
straight_velocity = 7

straight_area.append([1, 10])

curve_area.append([15, 40])

#############################################



if __name__ == "__main__":
    
    pid = []
    data = []

    straight_area_data = []
    buffer_area_data = []
    curve_area_data = []

    #####################
    # read point.csv file
    #####################
    with open(point_csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ",")
        line_counter = 0
        
        for row in csv_reader:
            if line_counter != 0:
                pid.append(float(row[0]))
                data.append([float(row[4]), float(row[5])])

            line_counter = line_counter + 1

    ########################
    # distinguish curve area
    ########################
    is_check = False

    for i in range(len(pid)):
        is_check = False

        for j in range(len(straight_area)):
            if ((i+1) >= straight_area[j][0]) & ((i+1) <= straight_area[j][1]):
                straight_area_data.append([data[i][0], data[i][1]])
                is_check = True
            
        for j in range(len(curve_area)):
            if ((i+1) >= curve_area[j][0]) & ((i+1) <= curve_area[j][1]):
                curve_area_data.append([data[i][0], data[i][1]])
                is_check = True
            
        if (is_check == False) :
            buffer_area_data.append([data[i][0], data[i][1]])
    
    straight_data = np.array(straight_area_data)
    curve_data = np.array(curve_area_data)
    buffer_data = np.array(buffer_area_data)
    
    #################
    # write yaml file
    #################
    dict_file = {
        "vel_setting" : {
            "straight_velocity" : straight_velocity,
            "buffer_velocity" : buffer_velocity,
            "curve_velocity" : curve_velocity,
            "use_algorithm" : use_algorithm
        }
    }

    with open(yaml_path, 'w') as file:
        documents = yaml.dump(dict_file, file)
        file.writelines("  straight_line_start: " + str(np.array(straight_area)[:, 0].tolist()) + "\n")
        file.writelines("  straight_line_end: " + str(np.array(straight_area)[:, 1].tolist()) + "\n")
        file.writelines("  curve_line_start: " + str(np.array(curve_area)[:, 0].tolist()) + "\n")
        file.writelines("  curve_line_end: " + str(np.array(curve_area)[:, 1].tolist()) + "\n")
        file.writelines("  way_points_x: " + str(np.array(data)[:, 0].tolist()) + "\n")
        file.writelines("  way_points_y: " + str(np.array(data)[:, 1].tolist()) + "\n")

    ##########################
    # visualize
    ##########################
    plt.figure(figsize=(6,10))
    plt.scatter(straight_data[:, 0], straight_data[:, 1], c="red")
    plt.scatter(curve_data[:, 0], curve_data[:, 1], c="blue")
    plt.scatter(buffer_data[:, 0], buffer_data[:, 1], c="green")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title("Scatter Plot")
    plt.show()
