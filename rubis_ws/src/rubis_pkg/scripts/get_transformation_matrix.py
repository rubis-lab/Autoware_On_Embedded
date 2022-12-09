import csv
import numpy as np
import math

### TODO ###
file_name = '2022-11-04_FMTC'
############

GNSS_X  = 0
GNSS_Y  = 1
NDT_X   = 2
NDT_Y   = 3


'''
ref: https://stackoverflow.com/questions/11687281/transformation-between-two-set-of-points

    [T11 T12 T13]
T = [T21 T22 T23]
    [T31 T32 T33] // We can ingnore T31, T32, T33 that (0, 0, 1)

[x'0]     [x0 y0 1  0  0  0 ]
[y'0]     [0  0  0  x0 y0 1 ]     [T11]
[x'1]     [x1 y1 1  0  0  0 ]     [T12]
[y'1]  =  [0  0  0  x1 y1 1 ]  *  [T13]
[x'2]     [x2 y2 1  0  0  0 ]     [T21]
[y'2]     [0  0  0  x2 y2 1 ]     [T22]
[x'3]     [x3 y3 1  0  0  0 ]     [T23]
[y'3]     [0  0  0  x3 y3 1 ]

P' = P_mod * T
T = pinv(P_mod) * P'

'''

def get_average_error(T_vec, P_mod, P_prime):
    prediction = np.matmul(P_mod, T_vec)

    return np.average(pow(P_prime - prediction,2))

def main():
    file_path = './data/'+file_name+'.csv'

    raw_data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)                
        for i, line in enumerate(reader):
            if i == 0: continue
            line_data = []
            for v in line: line_data.append(float(v))
            raw_data.append(line_data)

    # Filter stoped points(Filter first/last 10%)
    n = len(raw_data)
    raw_data = raw_data[int(n*0.1): int(n*0.9)]

    # Init data
    # P: Gnss pose / P_prime: Lidar pose
    P_mod_list = []
    P_prime_list = []
    
    for data in raw_data:
        P_mod_list.append([data[GNSS_X], data[GNSS_Y], 1, 0,            0,            0])
        P_mod_list.append([0,            0,            0, data[GNSS_X], data[GNSS_Y], 1])
        P_prime_list.append(data[NDT_X])
        P_prime_list.append(data[NDT_Y])
        
    
    P_mod = np.array(P_mod_list)
    P_prime = np.array(P_prime_list)    
    T_vec = np.matmul(np.linalg.pinv(P_mod), P_prime)

    T = np.array([[T_vec[0], T_vec[1], T_vec[2]],
                  [T_vec[3], T_vec[4], T_vec[5]],
                  [0,          0,          1        ]])
    
    print('T:', T)

    with open('../../../../autoware.ai/autoware_files/transformation/'+file_name+'.yaml', 'w+') as f:        
        f.write('gnss_transformation: ['+str(T[0,0])+', '+str(T[0,1])+', '+str(T[0,2])+', '+str(T[1,0])+', '+str(T[1,1])+', '+str(T[1,2])+', '+str(T[2,0])+', '+str(T[2,1])+', '+str(T[2,2])+']\n')
    
    print('Create transformation yaml file to -> autoware.ai/autoware_files/transformation/'+file_name+'.yaml')

    return

if __name__ == '__main__':
    main()