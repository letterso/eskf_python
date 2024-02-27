from eskf import ESKF, StaticIMUInit, IMUState
import utm
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import colorlog
import time
import threading
import signal
from queue import Queue

import matplotlib.style as mplstyle
mplstyle.use('fast')


class SignalHandler:
    """ 捕获ctrl+c 终止信号
        释放资源，防止中断程序造成异常
    """

    def __init__(self):
        self.run_enbale = False
        signal.signal(signal.SIGINT, self.exit_gracefully)  # 连接中断
        signal.signal(signal.SIGTERM, self.exit_gracefully)  # 终止

    def exit_gracefully(self, *args):
        self.run_enbale = True


predict_p_queue = Queue()
update_p_queue = Queue()
stop_event = threading.Event()
def plot_function():

    # gnss_plot, = plt.plot([], [], ".r")
    predit_plot_list = np.zeros((3, 500))

    # gnss_plot, = plt.plot([], [], ".r")
    gnss_plot_list = np.zeros((3, 500))
    # update_plot, = plt.plot([], [], ".g")
    update_plot_list = np.zeros((3, 500))
    while not stop_event.is_set():
        # T1 = time.time()
        plt.cla()
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        # plot_covariance_ellipse(xEst, PEst)
        # plt.xlim(-150, 150)
        # plt.ylim(-150, 150)

        # get predict data
        while not predict_p_queue.empty():
            predict_p = predict_p_queue.get(timeout=0.01)
            predit_plot_list[:, :-1] = predit_plot_list[:, 1:]
            predit_plot_list[:, -1][0], predit_plot_list[:, -1][1], predit_plot_list[:, -1][2] = predict_p

        # get update data
        while not update_p_queue.empty():
            update_p = update_p_queue.get(timeout=0.01)
            gnss_plot_list[:, :-1] = gnss_plot_list[:, 1:]
            update_plot_list[:, :-1] = update_plot_list[:, 1:]
            gnss_plot_list[:, -1][0], gnss_plot_list[:, -1][1], gnss_plot_list[:, -1][2], update_plot_list[:, -1][0], update_plot_list[:, -1][1], update_plot_list[:, -1][2] = update_p

        # gnss_plot.set_data(gnss_plot_list[0], gnss_plot_list[1])
        # update_plot.set_data(update_plot_list[0], update_plot_list[1])
        plt.plot(predit_plot_list[0], predit_plot_list[1], ".b")
        plt.plot(gnss_plot_list[0], gnss_plot_list[1], ".r")
        plt.plot(update_plot_list[0], update_plot_list[1], ".g")

        # draw
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)
        # T2 = time.time()
        # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    
    # 释放资源
    if predict_p_queue.empty():
        while not predict_p_queue.empty():
            predict_p_queue.get()
    if update_p_queue.empty():
        while not update_p_queue.empty():
            update_p_queue.get()


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
    TGB[0:3, 3:4] = - \
        TGB[0:3, 0:3] @ np.array([[antenna_pos[0]], [antenna_pos[1]], [0]])
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
    first_gnss = True
    origin_pose = np.zeros(3).reshape(-1, 1)
    last_gnss_pose = np.zeros(3).reshape(-1, 1)

    
    # 里程计参数
    odom_var = 0.5
    odom_span = 0.1        # 里程计测量间隔
    wheel_radius = 0.155   # 轮子半径
    circle_pulse = 1024.0  # 编码器每圈脉冲数

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
    eskf_filter = ESKF(logger)
    eskf_init = StaticIMUInit(logger)
    eskf_init_finish = False

    # 绘图
    if with_ui:
        plot_thread = threading.Thread(target=plot_function)
        plot_thread.daemon = True
        plot_thread.start()

    signal_handler = SignalHandler()
    with (Path.cwd() / 'data'/'10.txt').open('r') as file:
        for line in file:
            # 中断退出
            if signal_handler.run_enbale:
                break

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
                    options.gyro_var_ = np.sqrt(eskf_init.gyro_cov_)
                    options.acce_var_ = np.sqrt(eskf_init.acce_cov_)
                    eskf_filter.SetInitialConditions(
                        options, eskf_init.init_ba_, eskf_init.init_bg_, eskf_init.gravity_)
                    eskf_init_finish = True

            else:
                # 预测
                # Todo 在无GPS时，积分会漂移，增加ZUPT/轮速计约束
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
                        predict_p_queue.put((pred_P[0],pred_P[1],pred_P[2]))
                        if predict_p_queue.qsize() > 500:
                            predict_p_queue.get()

                    #     plt.plot(predict_p_plot[0], predict_p_plot[1], ".b")

                # 更新
                if data_items[0] == 'ODOM':
                    velo = np.zeros(2)
                    velo[0] = wheel_radius * float(data_items[2]) / circle_pulse * 2 * math.pi / odom_span
                    velo[1] = wheel_radius * float(data_items[3]) / circle_pulse * 2 * math.pi / odom_span
                    eskf_filter.ObserveWheelSpeed(velo,odom_var*odom_var)

                if data_items[0] == 'GNSS':
                    # 角度异常时不进行更新
                    if int(data_items[6]) != 1:
                        continue

                    # to utm
                    gnss_pos, gnss_r = ConvertGps2UTM(np.array([float(data_items[2]), float(data_items[3]), float(data_items[4]), float(data_items[5])]),  # lat lon alt heading
                                                      np.array([antenna_pox_x, antenna_pox_y]), antenna_angle)

                    if first_gnss:
                        origin_pose = gnss_pos
                        first_gnss = False
                        continue

                    # 移除起点
                    gnss_pos = gnss_pos - origin_pose
                    gnss_p_plot = gnss_pos

                    trans_noise = 0.1
                    ang_noise = np.deg2rad(1.0)
                    eskf_filter.ObserveSE3(
                        gnss_pos, gnss_r, trans_noise, ang_noise)

                    # eskf_filter.ObserveTran(
                    #     gnss_pos, trans_noise)
                    _, update_P, update_v, update_R, _, _ = eskf_filter.GetNominalState()
                    update_p_plot = update_P
                    logger.debug("update:\n p: {} \n v: {} \n r: {}".format(
                        update_P, update_v, update_R))

                    if with_ui:
                        if math.sqrt(np.sum(np.power(gnss_pos - last_gnss_pose,2))) >5:
                            update_p_queue.put((gnss_pos[0], gnss_pos[1], gnss_pos[2],
                                                update_P[0], update_P[1], update_P[2]))
                            if update_p_queue.qsize() > 100:
                                update_p_queue.get()
                            last_gnss_pose = gnss_pos

                time.sleep(0.001)

    if with_ui:
        stop_event.set()
        plot_thread.join()
