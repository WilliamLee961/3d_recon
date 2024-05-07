# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points
# 翻译：
# 使用极线几何来找到基本矩阵和本质矩阵，
# 使用本质矩阵来找到两个相机之间的相对姿态，
# 最后使用相对姿态来进行三维点的三角测量。

import cv2
import numpy as np
import os
from feature_extraction import extract_features
from feature_matching import match_features

def initialize_scene(image1_path, image2_path, camera_matrix):
    # step1: 获取匹配关键点
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    feature1 = extract_features([image1])
    feature2 = extract_features([image2])
    keypoints1 = feature1[0][0]
    keypoints2 = feature2[0][0]
    matches = match_features(feature1[0], feature2[0]) #这里与feature_matching对应，如果这里用feature1，那么match函数也要改

    # 提取匹配的特征点
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(matched_keypoints1, matched_keypoints2, cv2.FM_RANSAC)

    # 从基础矩阵中计算本质矩阵
    E = np.matmul(np.matmul(np.transpose(camera_matrix), F), camera_matrix)

    # 使用本质矩阵计算相对姿态
    _, R, t, mask = cv2.recoverPose(E, matched_keypoints1, matched_keypoints2)

    # 返回相对姿态和匹配的关键点
    return R, t, matched_keypoints1, matched_keypoints2

# 调用初始函数
current_path = os.getcwd()
folder = 'images'
img_folder = os.path.join(current_path, folder)

img_files = os.listdir(img_folder)
image_paths = [os.path.join(img_folder, file) for file in img_files]

# camera_matrix 2759.48 0 1520.69
# 0 2764.16 1006.81
# 0 0 1
camera_matrix = np.array([[2759.48, 0, 1520.69], [ 0, 2764.16, 1006.81], [0, 0, 1]])
image1_path = image_paths[0]
image2_path = image_paths[1]
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
feature1 = extract_features([image1])
feature2 = extract_features([image2])

R, t, matched_keypoints1, matched_keypoints2 = initialize_scene(image1_path, image2_path, camera_matrix)
print("R:", R)
print("t:", t)
print("matched keypoints:",matched_keypoints1)

# print(1111)
# print(image1_path)
