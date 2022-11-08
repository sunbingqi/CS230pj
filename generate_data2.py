import numpy as np
import random
import math
from matplotlib import pyplot as plt
from common import *
from PIL import Image


def homography_transform(X, H):
    # TODO
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)

    Y = H @ np.vstack((X.T, np.ones((1, X.shape[0]))))
    return (Y[:2] / Y[2].reshape(1, Y.shape[1])).T


def fit_homography(XY):
    # TODO
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    X = XY[:, :2]
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    Y = XY[:, 2:]
    A = np.zeros((X.shape[0]*2, 9))
    A[1::2, 0:3] = X
    A[::2, 3:6] = -1 * X
    A[::2, 6:] = Y[:, 1].reshape(X.shape[0], 1) * X
    A[1::2, 6:] = -Y[:, 0].reshape(X.shape[0], 1) * X
    val, vec = np.linalg.eig(A.T @ A)
    h = vec[:, np.argmin(val)]
    return h.reshape(3, 3) / h[-1]

# TODO: set staring number of name
name_staring = 0
# TODO: set range
data_range = 50

for index in range(data_range):
    #load image
    background_idx = random.randint(0,12)
    background = read_colorimg("./background/background"+str(background_idx)+".jpg")
    card_class1 = random.randint(1,13)
    all_type = ['c','d','h','s']
    card_type = random.randint(0,3)
    poker = read_colorimg("./poker/poker" + str(card_class1) + all_type[card_type] + ".JPG")
    card_class = random.randint(1,13)
    card_type = random.randint(0,3)
    poker2 = read_colorimg("./poker/poker" + str(card_class) + all_type[card_type] + ".JPG")

    #Random Contrast/brightness
    alpha = random.random()+0.5
    beta = int(random.random()*50)-25
    new_poker = np.clip((poker*alpha+beta).astype(int), 0, 255)
    alpha = random.random()+0.5
    beta = int(random.random()*50)-25
    new_poker2 = np.clip((poker2*alpha+beta).astype(int), 0, 255)

    #rotation
    angle = random.random()*2*math.pi
    H_rotate_temp = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    angle1 = random.random()*math.pi/4
    angle2 = random.random()*math.pi/4
    H_rotate_temp2 = np.array([[math.cos(angle1),0,0],[0,math.cos(angle2),0],[0,0,1]])
    H_rotate = np.dot(H_rotate_temp, H_rotate_temp2)
    angle = random.random()*2*math.pi
    H_rotate_temp = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    H_rotate2 = np.dot(H_rotate_temp, H_rotate_temp2)

    #scaling
    x_ratio = abs(background.shape[0] / (poker.shape[0]*abs(math.cos(angle))+poker.shape[1]*abs(math.sin(angle))))
    y_ratio = abs(background.shape[1] / (poker.shape[1]*abs(math.cos(angle))+poker.shape[0]*abs(math.sin(angle))))
    ratio = min(x_ratio, y_ratio)
    ratio = random.random()*ratio*10/20 + ratio*2/10
    H_scale = np.array([[ratio,0,0],[0,ratio,0],[0,0,1]])


    #translations
    temp_H = np.dot(H_scale, H_rotate)
    origin_point = np.array([[0, 0], [poker.shape[1], 0], [0, poker.shape[0]], [poker.shape[1], poker.shape[0]]])
    new_point = homography_transform(origin_point, temp_H)
    x_min = int(max(0, -np.min(new_point, 0)[1]))
    y_min = int(max(0, -np.min(new_point, 0)[0]))
    x_max = int(background.shape[0]-np.max(new_point, 0)[1])
    y_max = int(background.shape[1]-np.max(new_point, 0)[0])
    H_tran = np.array([[1,0,random.randint(y_min,y_max+1)],[0,1,random.randint(x_min,x_max+1)],[0,0,1]])
    final_H = np.dot(H_tran, temp_H)

    final_H[2][0] = 10**(-random.random()*3-5)
    final_H[2][1] = 10**(-random.random()*3-5)

    relate_x = int(poker2.shape[0] *ratio * 0.3)
    temp_H2 = np.dot(H_scale, H_rotate2)
    H_tran[0][2] += random.randint(relate_x,int(relate_x * 1.5))
    H_tran[1][2] += random.randint(relate_x,int(relate_x * 1.5))
    final_H2 = np.dot(H_tran, temp_H2)
    final_H2[2][0] = final_H[2][0]
    final_H2[2][1] = final_H[2][1]

    #get result
    final_card = cv2.warpPerspective(poker, final_H, (background.shape[1], background.shape[0]), flags=2)
    final_mid = final_card.copy()
    final_card2 = cv2.warpPerspective(poker2, final_H2, (background.shape[1], background.shape[0]), flags=2)
    final_mid[(np.sum(final_mid, 2) == 0)] = final_card2[(np.sum(final_mid, 2) == 0)]
    final_mid[(np.sum(final_mid, 2) == 0)] = background[(np.sum(final_mid, 2) == 0)]
    final_image = Image.fromarray(final_mid)
    


    #calculate center width/hight
    class_poker = card_class1 - 1
    origin_point = np.array([[0, 0], [poker.shape[1], 0], [0, poker.shape[0]], [poker.shape[1], poker.shape[0]]])
    new_point = homography_transform(origin_point, final_H)

    boundary_min = np.amin(new_point, axis=0)
    boundary_max = np.amax(new_point, axis=0)

    center_x = (boundary_max[0]+boundary_min[0])/(2 * background.shape[1])
    center_y = (boundary_max[1]+boundary_min[1])/(2 * background.shape[0])
    width = (boundary_max[0]-boundary_min[0])/background.shape[1]
    hight = (boundary_max[1]-boundary_min[1])/background.shape[0]

    class_poker2 = card_class - 1
    origin_point2 = np.array([[0, 0], [poker2.shape[1], 0], [0, poker2.shape[0]], [poker2.shape[1], poker2.shape[0]]])
    new_point2 = homography_transform(origin_point2, final_H2)

    boundary_min2 = np.amin(new_point2, axis=0)
    boundary_max2 = np.amax(new_point2, axis=0)

    if boundary_min2[0] < 0 or boundary_min2[1] < 0 or boundary_max2[0] > background.shape[1] or boundary_max2[1] > background.shape[0]:
        continue
    

    center_x2 = (boundary_max2[0]+boundary_min2[0])/(2 * background.shape[1])
    center_y2 = (boundary_max2[1]+boundary_min2[1])/(2 * background.shape[0])
    width2 = (boundary_max2[0]-boundary_min2[0])/background.shape[1]
    hight2 = (boundary_max2[1]-boundary_min2[1])/background.shape[0]
    with open('./mydataset_labels/data_'+str(name_staring+index)+'.txt', "a+") as f:
        f.write(str(class_poker)+' '+str(center_x)+' '+str(center_y)+' '+str(width)+' '+str(hight)+'\n')
        f.write(str(class_poker2)+' '+str(center_x2)+' '+str(center_y2)+' '+str(width2)+' '+str(hight2)+'\n')
    final_image.save("./mydataset_images/data_"+str(name_staring+index)+".jpg")
