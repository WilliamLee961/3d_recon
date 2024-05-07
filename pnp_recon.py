import cv2
import numpy as np
import os
from feature_extraction import extract_features
from feature_matching import match_features
from initial_recon import initialize_scene


# perform 3D reconstruction using PnP
def reconstruct_scene(image_paths, camera_matrix, R_initial, t_initial):
    points_3d = []
    R_prev, t_prev = R_initial, t_initial
    for image_path in image_paths:
        image = cv2.imread(image_path)
        features = extract_features([image])
        keypoints = features[0][0]
        # 图像与自身匹配
        matches = match_features(features[0], features[0])
        matched_keypoints = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        print(type(matched_keypoints))

        matched_keypoints = matched_keypoints.reshape(-1, 2)
        print(matched_keypoints.shape)
        # 检查匹配点的数量
        if len(matched_keypoints) < 4:
            print("Not enough matched keypoints for PnP. Skipping this image.")
            continue

        object_points = np.zeros((len(matched_keypoints), 3), dtype=np.float32)  # 生成形状为 (N, 3) 的数组
        # 通过PnP算法计算相机姿态和三维点坐标
        _, R, t, inliers = cv2.solvePnPRansac(objectPoints=object_points,
                                              imagePoints=matched_keypoints,
                                              cameraMatrix=camera_matrix,
                                              distCoeffs=None,
                                              flags=cv2.SOLVEPNP_EPNP)

        if R_prev is not None and t_prev is not None:
            R = np.dot(R_prev, R)
            t = np.dot(R_prev, t) + t_prev
        # 计算投影矩阵
        projection_matrix = np.hstack((R,t))
        print(projection_matrix.shape)  # 3*2,应该是3*4 
        # 计算三维点
        points_3d.append(cv2.triangulatePoints(projMatr1=np.eye(3,4),
                                               projMatr2=projection_matrix,
                                               projPoints1=matched_keypoints.T,
                                               projPoints2=matched_keypoints.T).reshape(-1,3))
        R_prev, t_prev = R,t

    points_3d = np.vstack(points_3d)
    return points_3d

# 测试重建场景：
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
R_initial, t_initial, _, _ = initialize_scene(image1_path, image2_path, camera_matrix)
print(R_initial,t_initial)

# 重建场景
points_3d = reconstruct_scene(image_paths, camera_matrix, R_initial, t_initial)

print("Reconstructed 3D points:", points_3d)
