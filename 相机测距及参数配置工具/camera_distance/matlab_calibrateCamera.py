# / -*-codeing = utf-8  -*-
# TIME : 2022/9/28 17:30
# File : matlab_calibrateCamera  

import cv2
import numpy as np

size = (640, 480)
# new para
# left_camera_matrix = np.array([[588.501481178388, 0., 342.531744984975],
#                                [0., 586.693976115446, 249.594574274635],
#                                [0., 0., 1.]])


# right_camera_matrix = np.array([[587.614620550390, 0., 334.228016785888],
#                                 [0., 586.367262716068, 248.250999983585],
#                                 [0., 0., 1.]])

# left_distortion = np.array([[0.262434016280846, -0.152673586679915, -0.0132549749288393, -0.00179859414286297, 0]])

# right_distortion = np.array([[0.243380883575201, 0.104353055196612, -0.0182350931472478, -0.00802771497646945, 0]])


# R = np.array([[0.999941704054268, 0.000704600006585645, 0.0107746012398446],
#               [-0.000731317600698892, 0.999996667432363, 0.00247594400901031],
#               [-0.0107728207825922, -0.00248367932703994, 0.999938886967292]])

# T = np.array([-60.0719984305519, 0.0661630540207382, 1.74410003401006])

# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix,
#                                                                                     left_distortion,
#                                                                                     right_camera_matrix,
#                                                                                     right_distortion, size,
#                                                                                     R, T)

# left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,
#                                                 left_distortion,
#                                                 R1, P1, size, cv2.CV_16SC2)

# right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,
#                                                     right_distortion,
#                                                     R2, P2, size, cv2.CV_16SC2)


#  old para

left_camera_matrix = np.array([[484.201664243347, 0., 339.348278626891],
                               [0., 482.587462519239, 268.927042884019],
                               [0., 0., 1.]])

right_camera_matrix = np.array([[486.046474458756, 0., 340.789777772544],
                                [0., 484.183934065625, 271.727267608496],
                                [0., 0., 1.]])

left_distortion = np.array([[0.118029855247932, -0.262364271043276, 0, 0, 0]])

right_distortion = np.array([[0.111804769563073, -0.0844424627176924, 0, 0, 0]])

R = np.array([[0.999977546210316, 0.000177066718322876, 0.00669893443567005],
              [-0.000195597191186026, 0.999996156514006, 0.00276562813031765],
              [-0.00669841898771223, -0.00276687632424482, 0.999973737444375]])

T = np.array([-60.1062899847100, -0.330137723076574, 1.81202328248923])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix,
                                                                  left_distortion,
                                                                  right_camera_matrix,
                                                                  right_distortion, size,
                                                                  R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix,
                                                   left_distortion,
                                                   R1, P1, size, cv2.CV_16SC2)

right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix,
                                                     right_distortion,
                                                     R2, P2, size, cv2.CV_16SC2)
