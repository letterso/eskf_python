from pathlib import Path
import logging
import colorlog
import math
import time

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

import utm
from eskf import ESKF, StaticIMUInit, IMUState

mplstyle.use('fast')

def ConvertGps2UTM(gps_msg, antenna_pos, antenna_angle):
    # pos
    utm_pos = np.zeros(3)
    utm_pos[0], utm_pos[1], _, _ = utm.from_latlon(
        gps_msg[0], gps_msg[1])
    utm_pos[2] = gps_msg[2]
    # heading
    utm_angle = np.deg2rad(90 - gps_msg[3])

    # TWG 转到 TWB
    TGB = np.identity(4)
    TGB[0:3, 0:3] = np.array([[np.cos(antenna_angle), -np.sin(antenna_angle), 0],
                              [np.sin(antenna_angle),  np.cos(
                                  antenna_angle), 0],
                              [0,           0, 1]]).T
    TGB[0:3, 3:4] = -TGB[0:3, 0:3] @ np.array([[antenna_pos[0]], [antenna_pos[1]], [0]])
    TWG = np.identity(4)
    TWG[0:3, 0:3] = np.array([[np.cos(utm_angle), -np.sin(utm_angle), 0],
                              [np.sin(utm_angle),  np.cos(utm_angle), 0],
                              [0,           0, 1]])
    TWG[0:3, 3:4] = np.array([[utm_pos[0]], [utm_pos[1]], [utm_pos[2]]])
    TWB = TWG @ TGB

    return TWB[0:3, 3:4], TWB[0:3, 0:3]


if __name__ == '__main__':
    # RTK天线安装偏移
    antenna_angle = np.deg2rad(12.06)
    antenna_pox_x = -0.17
    antenna_pox_y = -0.20
    first_gps = True
    origin_pose = np.zeros(3).reshape(-1, 1)

    # 图形界面
    with_ui = True
    predict_p_plot = np.zeros((3, 1))
    gnss_p_plot = np.zeros((3, 1))
    update_p_plot = np.zeros((3, 1))

    # 日志
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    # 创建文件日志处理器
    file_handler = logging.FileHandler(
        str(Path.cwd() / 'output'/'log.txt'), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    file_handler.setLevel(logging.WARN)
    # 移除默认的handler
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # 将日志处理器添加到logger对象
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # ESKF
    eskf_filter = ESKF()
    eskf_init = StaticIMUInit(logger)
    eskf_init_finish = False

    with (Path.cwd() / 'data'/'10.txt').open('r') as file:
        for line in file:
            # 数据获取
            line = line.strip()  # 去除行末的换行符和空白字符
            data_items = line.split()  # 使用空格分隔数据项

            if not eskf_init_finish:
                # 初始化
                if not eskf_init.InitSuccess():
                    if data_items[0] == 'IMU':
                        gyro = np.array([float(data_items[2]), float(
                            data_items[3]), float(data_items[4])])
                        acce = np.array([float(data_items[5]), float(
                            data_items[6]), float(data_items[7])])
                        imu_data = IMUState(float(data_items[1]), gyro, acce)
                        eskf_init.AddIMU(imu_data)

                if eskf_init.InitSuccess():
                    options = ESKF.Options()
                    options.gyro_var_ = math.sqrt(eskf_init.gyro_cov_[0])
                    options.acce_var_ = math.sqrt(eskf_init.acce_cov_[0])
                    eskf_filter.SetInitialConditions(
                        options, eskf_init.init_ba_, eskf_init.init_bg_, eskf_init.gravity_)
                    eskf_init_finish = True

            else:
                # 预测
                if data_items[0] == 'IMU':
                    gyro = np.array([float(data_items[2]), float(
                        data_items[3]), float(data_items[4])])
                    acce = np.array([float(data_items[5]), float(
                        data_items[6]), float(data_items[7])])
                    imu_data = IMUState(float(data_items[1]), gyro, acce)
                    eskf_filter.Predict(imu_data)
                    _, pred_P, pred_v, pred_R, _, _ = eskf_filter.GetNominalState()
                    predict_p_plot = pred_P
                    logger.debug("predict:\n p: {} \n v: {} \n r: {}".format(
                        pred_P, pred_v, pred_R))

                    if with_ui:
                        plt.plot(predict_p_plot[0], predict_p_plot[1], ".b")

                # 更新
                if data_items[0] == 'GNSS':

                    gnss_pos, gnss_r = ConvertGps2UTM(np.array([float(data_items[2]), float(data_items[3]), float(data_items[4]), float(data_items[5])]),
                                                            np.array([antenna_pox_x, antenna_pox_y]),
                                                            antenna_angle)  # lat lon alt heading

                    if first_gps:
                        origin_pose = gnss_pos
                        first_gps = False
                        continue

                    # 移除起点
                    gnss_pos = gnss_pos - origin_pose
                    gnss_p_plot = gnss_pos

                    trans_noise = 0.1
                    ang_noise = np.deg2rad(1.0)
                    eskf_filter.ObserveSE3(
                        gnss_pos, gnss_r, trans_noise, ang_noise)
                    _, update_P, update_v, update_R, _, _ = eskf_filter.GetNominalState()
                    update_p_plot = update_P
                    logger.debug("update:\n p: {} \n v: {} \n r: {}".format(
                        update_P, update_v, update_R))

                    if with_ui:
                        # plt.cla()
                        # for stopping simulation with the esc key.
                        # plt.gcf().canvas.mpl_connect('key_release_event',
                        #         lambda event: [exit(0) if event.key == 'escape' else None])
                        # plt.plot(predict_p_plot[0], predict_p_plot[1], ".b")
                        plt.plot(gnss_p_plot[0], gnss_p_plot[1], ".r")
                        plt.plot(update_p_plot[0], update_p_plot[1], ".g")
                        # plt.plot(hxTrue[0, :].flatten(),
                        #         hxTrue[1, :].flatten(), "-b")
                        # plt.plot(hxDR[0, :].flatten(),
                        #         hxDR[1, :].flatten(), "-k")
                        # plt.plot(hxEst[0, :].flatten(),
                        #         hxEst[1, :].flatten(), "-r")
                        # plot_covariance_ellipse(xEst, PEst)
                        # plt.xlim(-150, 150)
                        # plt.ylim(-150, 150)
                        plt.axis("equal")
                        plt.grid(True)
                        plt.pause(0.0001)

                # time.sleep(0.001)
