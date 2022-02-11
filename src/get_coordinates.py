import os
import numpy as np
from detector import detector

def get_all_camera_coordinate(path):
    
    folder_paths = os.listdir(path)
    # print(folder_paths)
    # angles transformation for [middle, west, east, south]
    angles = [0.115192,0.115192,0.0, 1.5708*3]
    # translational tranformation
    X_tran = [273.56, 2*248.23, 0, 273.56]
    Y_trans = [0, 0, 0, 273.56]

    all_camera_list = []

    for i in range(0,len(folder_paths)):
        print(folder_paths[i])
        path_folders = path + folder_paths[i]
        # print(path_folders)
        files_list = os.listdir(path_folders)
        # print(files_list)
        each_camera = []
        for files in files_list:
            file_path = path_folders + '/' + files
            X_coordinates,y_coordinates= detector(file_path)
            # print(X_coordinates, y_coordinates)
            X_new = X_coordinates*np.cos(angles[i]) -y_coordinates*np.sin(angles[i])+ X_tran[i]
            Y_new = X_coordinates*np.sin(angles[i]) + y_coordinates*np.cos(angles[i])+Y_trans[i]
            each_camera.append([X_new,Y_new])
        all_camera_list.append(each_camera)
    return all_camera_list

# shaping the list to get a list of all camera measurements at each timestep.   
def get_list_for_filter(x, camera_readings):
  main = []
  for i in range(0,len(x[0])):
    all = []
    for j in range(0,camera_readings):
      print(x[j][i])
      all.append(list(map(abs,x[j][i])))
    main.append(all)
  print(main)
  return main    


# number = 2
# path =  f"C:/Users/faiza/Downloads/AMR_exercise_10_12_solution/AMR_exercise_10_12_solution/camera-{number}/"
# list_ = get_all_camera_coordinate(path)
# # pprint.pprint(list_)
# list = get_list_for_filter(list_)
# pprint.pprint(list)