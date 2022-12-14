import SPMUtil as spmu
import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy import stats
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import Kmean as kmean





def feature_matching(data1_path, data2_path, flatten=spmu.FlattenMode.Average, draw_plot=False):
    """
    compare two image to calculate the pixel shift in x, y, z
    :param data1_path:previous data serializer path
    :param data2_path:present data serializer path
    :param flatten:flatten method in spmutil
    :param draw_plot:write image
    :return: dx(x pixel shift), x_std, dy, y_std, dz, sec(time duration in second)
    """
    if data1_path is bytes:
        data1_path = data1_path.encode()
    if data2_path is bytes:
        data2_path = data2_path.encode()
    data1 = spmu.DataSerializer(data1_path)
    data1.load()
    data2 = spmu.DataSerializer(data2_path)
    data2.load()

    map1 = np.asarray(data1.data_dict['FWFW_ZMap'])
    map2 = np.asarray(data2.data_dict['FWFW_ZMap'])

    if flatten == spmu.FlattenMode.Average or spmu.FlattenMode.Off:
        map1 = spmu.flatten_map(map1, flatten)
        map2 = spmu.flatten_map(map2, flatten)
    else:
        flatten_param = spmu.get_flatten_param(map1, flatten, poly_fit_order=3)
        map1 = spmu.apply_flatten_plane(map1, *flatten_param, poly_fit_order=3)
        map2 = spmu.apply_flatten_plane(map2, *flatten_param, poly_fit_order=3)

    map1 = spmu.filter_2d.GaussianHannMap(map1, 11, 1, 1)
    map2 = spmu.filter_2d.GaussianHannMap(map2, 11, 1, 1)


    source_map = map1
    matching_map = map2


    detector = cv2.AKAZE_create()

    matching_map = matching_map - np.min(matching_map)
    source_map = source_map - np.min(source_map)
    im_source = (source_map / np.max(source_map) * 255).astype(np.uint8)
    im_search = (matching_map / np.max(matching_map) * 255).astype(np.uint8)

    kp1, des1 = detector.detectAndCompute(im_source, None)
    kp2, des2 = detector.detectAndCompute(im_search, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # keypoint_img1 = cv2.drawKeypoints(im_source, kp1, None, flags=0)
    # keypoint_img2 = cv2.drawKeypoints(im_search, kp2, None, flags=0)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    movement_list_x = []
    movement_list_y = []
    good_matches = []
    n = min(len(matches) - 1, 2)
    good_distance = np.mean([it.distance for it in matches[:n]]) * 2
    for mat in matches[:10]:
        if mat.distance < good_distance:
            good_matches.append(mat)
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            movement_list_x.append(x1 - x2)
            movement_list_y.append(y1 - y2)

    if len(movement_list_x) > 0:
        if len(movement_list_x) >= 5:
            pts_list, error, k = kmean.KmeanAutoK(np.array([movement_list_x, movement_list_y]).transpose(), distance=10)
            pts_list = sorted(pts_list, key=lambda x: -len(x))
            pts = np.array(pts_list[0])
            if len(pts) > len(movement_list_x) / 2 - 1:
                movement_list_x = pts[:, 0]
                movement_list_y = pts[:, 1]
        else:
            pts = np.array([movement_list_x, movement_list_y])
            print("warning: not enough matched points. match result may be in low accuracy.",
                  len(movement_list_x), "detected.")

        data2_header = spmu.ScanDataHeader.from_dataSerilizer(data2)
        sec = data2_header.End_Scan_Sec

        if draw_plot:
            selected_good_match = []
            for it in good_matches:
                (x1, y1) = kp1[it.queryIdx].pt
                (x2, y2) = kp2[it.trainIdx].pt
                if np.min(pts[:, 0]) <= x1 - x2 <= np.max(pts[:, 0]):
                    selected_good_match.append(it)

            def enlarge(img: np.ndarray):
                s = img.shape
                m = np.ones(shape=(260, 260), dtype=img.dtype) * 255
                m[:s[0], :s[1]] = img
                return m
            img3 = cv2.drawMatches(enlarge(im_source), kp1, enlarge(im_search), kp2, good_matches, None, matchColor=(0, 0, 255),
                                   flags=cv2.DrawMatchesFlags_DEFAULT)
            img4 = cv2.drawMatches(np.zeros((260, 260, 3), np.uint8), kp1,
                                   np.zeros((260, 260, 3), np.uint8), kp2,
                                   selected_good_match, None, matchColor=(0, 255, 0),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.basename(data2.path) + "_1.jpg", img3+img4)

        dx, dy, dz = round(float(np.mean(movement_list_x))), round(float(np.mean(movement_list_y))), 0
        data1.load()
        data2.load()
        map2 = np.asarray(data2.data_dict['FWFW_ZMap'])
        map1 = np.asarray(data1.data_dict['FWFW_ZMap'])

        dxs, dys = 1, 1
        if dx == dxs or dx == -dxs:
            dxs = 2
        if dy == dys or dy == -dys:
            dys = 2

        if dx >= 0 and dy >= 0:
            z_drift_map = map2[dys:-dy-dys, dxs:-dx-dxs] - map1[dy+dys:-dys, dx+dxs:-dxs]
        elif dx >= 0 and dy < 0:
            z_drift_map = map2[dys-dy:-dys, dxs:-dx-dxs] - map1[dys:dy-dys, dx+dxs:-dxs]
        elif dx < 0 and dy >= 0:
            z_drift_map = map2[dys:-dy-dys, dxs-dx:-dxs] - map1[dys+dy:-dys, dxs:dx-dxs]
        elif dx < 0 and dy > 0:
            z_drift_map = map2[dys-dy:-dys, dxs-dx:-dxs] - map1[dys:dy-dys, dxs:dx-dxs]
        else:
            z_drift_map = np.zeros(1)
        dz = stats.trim_mean(np.ndarray.flatten(z_drift_map), 0.25)
        print("dz=", dz, "±", np.std(z_drift_map))
        print("dx=", float(np.mean(movement_list_x)), "±", np.std(movement_list_x),
              "dy=", float(np.mean(movement_list_y)), "±", np.std(movement_list_y))
        return float(np.mean(movement_list_x)), np.std(movement_list_x), \
               float(np.mean(movement_list_y)), np.std(movement_list_y), dz, sec
    else:
        print("no feature matched...")
        return None, None, None, None, None, None





def feature_point_matching_task(dataSerializers):
    """
    Calculate drift speed from a sequence data
    :param dataSerializers:list of dataSerializer
    :return px, py, py(drift speed -- in our scan system, the unit is Voltage/s)
    """
    def func(x, a, b):
        f = a * x + b
        return f
    time_stamp_list = []
    dx_list = []
    dy_list = []
    dz_list = []

    data = dataSerializers[0]
    param = spmu.PythonScanParam().from_dataSerilizer(data)
    header = spmu.ScanDataHeader().from_dataSerilizer(data)

    for i in range(0, len(dataSerializers) - 1):
        dx, dy, dz, sec = feature_matching(dataSerializers[0].path, dataSerializers[i + 1].path)
        if dx is None:
            print("drift estimation stop")
            return

        if i == 0:
            time_stamp_list.append(header.Start_Scan_Timestamp)
            dx_list.append(0)
            dy_list.append(0)
            dz_list.append(0)

        time_stamp_list.append(sec)
        dx_list.append(dx)
        dy_list.append(dy)
        dz_list.append(dz)

        duration = sec - header.Start_Scan_Timestamp
        print(duration, dx * param.Aux1DeltaVoltage / duration, dy * param.Aux2DeltaVoltage / duration)

    time_stamp_list = np.array(time_stamp_list)
    time_stamp_list -= np.min(time_stamp_list)
    dx_list = np.array(dx_list) * param.Aux1DeltaVoltage
    dy_list = np.array(dy_list) * param.Aux1DeltaVoltage
    dz_list = np.array(dz_list)

    px, _ = curve_fit(func, time_stamp_list, dx_list)
    py, _ = curve_fit(func, time_stamp_list, dy_list)
    pz, _ = curve_fit(func, time_stamp_list, dz_list)
    rot_z = param.ZRotation / 180 * np.pi
    mat = np.array([[np.cos(rot_z), -np.sin(rot_z)],
                    [np.sin(rot_z), np.cos(rot_z)]])
    px, py = np.matmul(mat, [px, py])
    return px, py, pz


