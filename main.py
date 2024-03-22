import cv2
import numpy as np
import os
import time  # 导入time模块


def process_frame(frame):
    """
    对每帧图像进行预处理。
    :param frame: 原始图像帧。
    :return: 预处理后的图像帧。
    """
    # 灰度化处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def extract_and_match_features(frame1, frame2, vis=False):
    """
    提取特征点并匹配。
    :param frame1: 第一帧图像。
    :param frame2: 第二帧图像。
    :return: 匹配点的坐标。
    """
    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测ORB特征并计算描述子
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # 初始化FLANN匹配器
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比率测试
    """
    对于每个点，如果最近邻距离与次近邻距离的比值小于某个阈值（例如0.75），则认为这是一个好的匹配。
    这种方法基于这样一个假设：好的匹配对之间的距离应该明显小于最近的错误匹配的距离。
    """
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:  # 确保每个匹配都有两个邻近点
            m, n = m_n
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 可视化
    if vis == True:
        for kp in kp2:
            x, y = kp.pt
            cv2.circle(frame, (int(x), int(y)), 3,
                       (255, 0, 0), -1)  # 用蓝色圆点标记特征点

    return src_pts, dst_pts


def estimate_motion(src_pts, dst_pts, K):
    """
    使用RANSAC算法计算运动估计。
    :param src_pts: 源图像中匹配点的坐标。
    :param dst_pts: 目标图像中匹配点的坐标。
    :param K: 相机内参矩阵。
    :return: R（旋转矩阵），t（平移向量）。
    """
    E, mask = cv2.findEssentialMat(
        src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
    return R, t


def update_pose_and_project(P, R, t, K):
    """
    更新相机位姿并计算投影矩阵。
    :param P: 当前的位姿矩阵。
    :param R: 新的旋转矩阵。
    :param t: 新的平移向量。
    :param K: 相机内参矩阵。
    :return: 更新后的位姿矩阵，第一帧的投影矩阵，第二帧的投影矩阵。
    """
    # 构建变换矩阵T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()

    # 更新位姿
    P_new = np.dot(P, np.linalg.inv(T))

    # 第一帧的投影矩阵（使用相机内参K和初始位姿）
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    # 第二帧的投影矩阵（使用相机内参K和当前位姿）
    P2 = np.dot(K, P_new[:3, :])

    return P_new, P1, P2


def triangulate_points(P1, P2, src_pts, dst_pts):
    """
    对匹配点进行三角测量以恢复三维坐标。
    :param P1: 第一帧的投影矩阵。
    :param P2: 第二帧的投影矩阵。
    :param src_pts: 源图像中匹配点的坐标。
    :param dst_pts: 目标图像中匹配点的坐标。
    :return: 三维点坐标。
    """
    # 确保点坐标是正确的形状
    src_pts = src_pts.reshape(-1, 2).T
    dst_pts = dst_pts.reshape(-1, 2).T

    # 三角测量
    points_4d_hom = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)

    # 将齐次坐标转换为3D坐标
    points_3d = points_4d_hom[:3] / points_4d_hom[3]

    # print("points_3d:", points_3d.shape)

    return points_3d.T


if __name__ == '__main__':
    # 设置相机参数
    F = int(os.getenv("F", "500"))
    W, H = 1920//2, 1080//2
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

    # 初始化位姿矩阵为单位矩阵，表示在世界坐标系原点
    P = np.eye(4)
    points_3d_all = []
    camera_positions = []  # 用于存储相机位置的列表

    # 初始化计时器和帧计数器
    start_time = time.time()
    frame_count = 0

    # 以视频文件或摄像头为例
    # cap = cv2.VideoCapture(0)  # 使用0代表摄像头，或者替换为视频文件路径
    # 替换为视频文件路径
    cap = cv2.VideoCapture('test.mp4')

    ret, prev_frame = cap.read()
    prev_frame = process_frame(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        src_pts, dst_pts = extract_and_match_features(
            prev_frame, frame, vis=True)
        R, t = estimate_motion(src_pts, dst_pts, K)

        # 更新位姿并获取投影矩阵
        P, P1, P2 = update_pose_and_project(P, R, t, K)
        camera_position = P[:3, 3]
        camera_positions.append(camera_position.tolist())  # 将位置添加到列表中

        # 三角测量恢复三维坐标
        points_3d = triangulate_points(P1, P2, src_pts, dst_pts)
        points_3d_all.append(points_3d)  # 可以选择如何处理或存储这些点

        # 可视化：
        cv2.imshow('Feature Matching', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame
        frame_count += 1  # 更新帧计数器

    cap.release()
    cv2.destroyAllWindows()

    # 计算并显示FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > 0:  # 防止除以零
        fps = frame_count / elapsed_time
        print(
            f"Processed {frame_count} frames in {elapsed_time:.2f} seconds, resulting in {fps:.2f} FPS.")
    else:
        print("No frames were processed.")
