import matplotlib.pyplot as plt
import numpy as np
import cv2
import SPMUtil as spmu


def get_align_rect(pts, margin=5, minimal_box_size=50):
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    xbox = max_x - min_x
    ybox = max_y - min_y
    margin = max(margin, max((minimal_box_size-xbox)/2, (minimal_box_size-ybox)/2))

    return spmu.Rect2D(points=(min_x - margin, min_y - margin), xbox=max_x - min_x + margin * 2,
                       ybox=max_y - min_y + margin * 2)


def Kmean(k, sample):
    Z = np.float32(sample)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    pts_list = []
    for i in range(k):
        A = Z[label.ravel() == i]
        pts_list.append(A)
    return pts_list, ret / sample.shape[0]


def plot_kmean_result(k, sample, feature_point_map=None):
    if feature_point_map is not None:
        plt.imshow(feature_point_map)
    Z = np.float32(sample)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    pts_list = []
    for i in range(k):
        A = Z[label.ravel() == i]
        pts_list.append(A)
    for it in pts_list:
        plt.scatter(it[:, 0], it[:, 1])
    plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
    plt.show()



def KmeanAutoK(sample, distance=625, inital_k=2):
    if len(sample) <= 3:
        raise ValueError("not enough feature point detected. (need at least 4 but " + str(len(sample)) +" detected)")

    pts_list, dist = Kmean(inital_k, sample)
    if dist <= distance:
        return pts_list, dist, inital_k
    else:
        return KmeanAutoK(sample, distance, inital_k+1)


def find_meet_pts_list(pts_list, pts_count=3, min_box_size=40, max_box_size=100):
    result_list = []
    pts_list = sorted(pts_list, key=lambda x: -len(x))
    condition = [(lambda x:len(x) > pts_count)
                 and (lambda x:max_box_size > max(max(x[:, 0]) - min(x[:, 0]), max(it[:, 1]) - min(it[:, 1])) > min_box_size)]
    for it in pts_list:
        if all(con(it) for con in condition):
            # print(max(it[:, 0]) - min(it[:, 0]), max(it[:, 1]) - min(it[:, 1]))
            result_list.append(it)
    if len(result_list) == 0:
        return pts_list[0]
    return result_list


def key_points_to_pts_list(kps, kmean_distance=625, pts_count=3):
    pts = np.array([it.pt for it in kps])
    pts_list, error, _ = KmeanAutoK(pts, distance=kmean_distance)
    pts_list = find_meet_pts_list(pts_list, pts_count=pts_count)
    return pts_list



if __name__ == '__main__':
    pass