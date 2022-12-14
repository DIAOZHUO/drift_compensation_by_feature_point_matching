import SPMUtil as spmu
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scripts.feature_matching import feature_matching, feature_point_matching_task
import os


# A1
file_list = [
    "./test_data/107_20220124_si110_drift.pkl",
    "./test_data/108_20220124_si110_drift.pkl",
    "./test_data/109_20220124_si110_drift.pkl"
]



time_stamp_list = []
dx_list = []
ex_list = []
dy_list = []
ey_list = []
dz_list = []
for i in range(len(file_list) - 1):
    dx, ex, dy, ey, dz, sec = feature_matching(file_list[0], file_list[i + 1], draw_plot=True)
    data = spmu.DataSerializer(file_list[0])
    data.load()
    param = spmu.PythonScanParam().from_dataSerilizer(data)
    header = spmu.ScanDataHeader().from_dataSerilizer(data)
    duration = sec - header.End_Scan_Sec
    if i == 0:
        time_stamp_list.append(header.End_Scan_Sec)
        dx_list.append(0)
        dy_list.append(0)
        dz_list.append(0)
        ex_list.append(0)
        ey_list.append(0)
    time_stamp_list.append(sec)
    dx_list.append(dx)
    dy_list.append(dy)
    dz_list.append(dz)
    ex_list.append(ex)
    ey_list.append(ey)




time_stamp_list = np.array(time_stamp_list)
dx_list = np.array(dx_list) * param.Aux1DeltaVoltage
dy_list = np.array(dy_list) * param.Aux1DeltaVoltage
ex_list = np.array(ex_list) * param.Aux1DeltaVoltage
ey_list = np.array(ey_list) * param.Aux1DeltaVoltage
dz_list = np.array(dz_list)


def func(x, a, b):
    f = a * x + b
    return f


px, x_cov = curve_fit(func, time_stamp_list, dx_list)
py, y_cov = curve_fit(func, time_stamp_list, dy_list)
pz, z_cov = curve_fit(func, time_stamp_list, dz_list)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.2))
axes[0].set_title("drift x")
axes[0].errorbar(time_stamp_list, dx_list, ex_list, linestyle='None', marker='^')
axes[0].plot(time_stamp_list, func(time_stamp_list, *px))

axes[1].set_title("drift y")
axes[1].errorbar(time_stamp_list, dy_list, ey_list, linestyle='None', marker='^')
axes[1].plot(time_stamp_list, func(time_stamp_list, *py))

axes[2].set_title("drift z")
axes[2].scatter(time_stamp_list, dz_list)
axes[2].plot(time_stamp_list, func(time_stamp_list, *pz))

axes[0].set_ylabel("drift speed (V/s)")
axes[0].set_xlabel("timestamp(s)")
axes[1].set_xlabel("timestamp(s)")
axes[2].set_xlabel("timestamp(s)")
plt.show()
print(px[0], py[0], pz[0])


