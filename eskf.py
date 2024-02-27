import numpy as np
import collections
from liegroups.liegroups.numpy import SO3, SE3


class IMUState():
    """ IMU数据
    """

    def __init__(self, timestamp: float, gyro: np.array, acce: np.array) -> None:
        self.timestamp_ = timestamp
        self.gyro_ = gyro
        self.acce_ = acce


class StaticIMUInit():
    def __init__(self, logger):
        """ 系统初始化
        """
        # 日志
        self.logger = logger

        # 初始化判断标志位
        self.init_success = False

        # 数据缓存
        self.imu_deque_ = collections.deque()

        # 参数
        self.init_time_seconds_ = 2             # 初始化需要采集时间
        self.init_imu_queue_max_size_ = 2000    # IMU数据缓存
        self.max_static_gyro_var_ = 0.5        # 静态下陀螺测量方差
        self.max_static_acce_var_ = 0.05       # 静态下加计测量方差
        self.gravity_norm_ = 9.81              # 重力大小

        # 零偏和噪声
        self.gravity_ = np.zeros(3)
        self.init_bg_ = 0.0
        self.init_ba_ = 0.0
        self.gyro_cov_ = np.zeros(3)
        self.acce_cov_ = np.zeros(3)

    def AddIMU(self, imu_data: IMUState):
        self.imu_deque_.append(imu_data)
        if len(self.imu_deque_) > self.init_imu_queue_max_size_:
            self.imu_deque_.popleft()

        init_time = imu_data.timestamp_ - self.imu_deque_[0].timestamp_
        if init_time > self.init_time_seconds_:
            self.TryInit()

    def TryInit(self):
        """ 进行初始化操作
        """

        # 获取原始数据
        imu_gyro_raw = np.zeros([len(self.imu_deque_), 3])
        imu_acce_raw = np.zeros([len(self.imu_deque_), 3])
        count = 0
        for imu_data in self.imu_deque_:
            imu_gyro_raw[count] = imu_data.gyro_
            imu_acce_raw[count] = imu_data.acce_
            count += 1

        # 计算均值和方差
        gyro_mean = np.mean(imu_gyro_raw, axis=0)
        self.gyro_cov_ = np.var(imu_gyro_raw, axis=0)

        acce_mean = np.mean(imu_acce_raw, axis=0)
        self.gravity_ = -acce_mean / \
            np.linalg.norm(acce_mean) * self.gravity_norm_
        imu_acce = imu_acce_raw+self.gravity_
        acce_mean = np.mean(imu_acce, axis=0)
        self.acce_cov_ = np.var(imu_acce, axis=0)

        # 检查IMU噪声
        if np.linalg.norm(gyro_mean) > self.max_static_gyro_var_:
            self.logger.warning("gyro noise too big {}".format(
                np.linalg.norm(gyro_mean)))
            self.init_success = False
            return

        if np.linalg.norm(acce_mean) > self.max_static_acce_var_:
            self.logger.warning("acce noise too big {}".format(
                np.linalg.norm(acce_mean)))
            self.init_success = False
            return

        # 估计测量噪声和零偏
        self.init_bg_ = gyro_mean
        self.init_ba_ = acce_mean
        self.logger.info("IMU init successful")
        self.logger.info("bg = {}, gyro sq = {}".format(
            self.init_bg_, self.gyro_cov_))
        self.logger.info("ba = {}, acce sq = {}".format(
            self.init_ba_, self.acce_cov_))
        self.logger.info("grav = {}, norm = {}".format(
            self.gravity_, np.linalg.norm(self.gravity_)))
        self.init_success = True

    def InitSuccess(self) -> bool:
        return self.init_success


class ESKF:
    def __init__(self, logger) -> None:
        # 日志
        self.logger = logger

        # 历史信息
        self.last_imu_ = IMUState(0, np.array([0, 0, 0]), np.array([0, 0, 0]))

        # State: p, v, q, ba, bg, g
        # 名义状态
        self.P_ = np.zeros(3).reshape(-1, 1)    # 3x1
        self.v_ = np.zeros(3).reshape(-1, 1)    # 3x1
        self.R_ = SO3(np.identity(3))
        self.ba_ = np.zeros(3).reshape(-1, 1)   # 3x1
        self.bg_ = np.zeros(3).reshape(-1, 1)   # 3x1
        self.g_ = np.array([[0], [0], [-9.8]])  # 3x1

        # 误差状态
        self.dx_ = np.zeros(18).reshape(-1, 1)

        # 协方差阵
        self.cov_ = np.identity(18)

        # 观测参数
        self.gnss_pos_noise_ = 0.1
        self.gnss_height_noise_ = 0.1
        self.gnss_ang_noise_ = np.deg2rad(1.0)

        self.options_ = self.Options()

    class Options():
        def __init__(self) -> None:
            # IMU参数
            self.imu_dt_ = 0.01                             # 频率
            self.gyro_var_ = np.array([1e-5, 1e-5, 1e-5])   # 磁力计噪声
            self.acce_var_ = np.array([1e-2, 1e-2, 1e-2])   # 加速计噪声
            self.bias_gyro_var_ = 1e-6                      # 磁力计零飘
            self.bias_acce_var_ = 1e-4                      # 加速计零飘

            # 更新参数
            self.update_bias_gyro_ = True
            self.update_bias_acce_ = True

    def GetNominalState(self):
        return self.last_imu_, self.P_, self.v_, self.R_, self.ba_, self.bg_

    def SetInitialConditions(self, options: Options, init_ba: float, init_bg: float, gravity: float):
        # 构建噪声
        self.BuildProcessNoise(options)
        self.options_ = options

        # 更新初始状态
        self.bg_ = init_bg.reshape(-1, 1)
        self.ba_ = init_ba.reshape(-1, 1)
        self.g_ = gravity.reshape(-1, 1)
        self.cov_ = np.identity(18) * 1e-4

    def BuildProcessNoise(self, options: Options):
        # 过程噪声
        # [P, V, R, Ba, Bg, g]
        ev2 = np.zeros(3)
        ev2[0] = options.acce_var_[0]  # * options.acce_var_[0]
        ev2[1] = options.acce_var_[1]  # * options.acce_var_[1]
        ev2[2] = options.acce_var_[2]  # * options.acce_var_[2]
        et2 = np.zeros(3)
        et2[0] = options.gyro_var_[0]  # * options.gyro_var_[0]
        et2[1] = options.gyro_var_[1]  # * options.gyro_var_[1]
        et2[2] = options.gyro_var_[2]  # * options.gyro_var_[2]
        eg2 = options.bias_gyro_var_  # * options.bias_gyro_var_
        ea2 = options.bias_acce_var_  # * options.bias_acce_var_

        self.Q_ = np.diag([0, 0, 0,
                           ev2[0], ev2[1], ev2[2],
                           et2[0], et2[1], et2[2],
                           eg2, eg2, eg2,
                           ea2, ea2, ea2,
                           0, 0, 0])

    def BuildMeasurementNoise(self, options: Options):
        # 测量噪声
        # [P,R]
        gp2 = self.gnss_pos_noise_ * self.gnss_pos_noise_
        gh2 = self.gnss_height_noise_ * self.gnss_height_noise_
        ga2 = self.gnss_ang_noise_ * self.gnss_ang_noise_
        self.gnss_noise_ = np.diag([gp2, gp2, gh2,
                                    ga2, ga2, ga2])

    def Predict(self, imu: IMUState) -> bool:
        # 检查数据
        dt = imu.timestamp_ - self.last_imu_.timestamp_
        if dt > 5 * self.options_.imu_dt_ or dt < 0:
            self.last_imu_ = imu
            return False

        # 名义递推,(3.41)
        new_p = self.P_ + self.v_ * dt + 0.5 * \
            (self.R_.as_matrix() @ (imu.acce_.reshape(-1, 1) - self.ba_)) * \
            dt * dt + 0.5 * self.g_ * dt * dt
        new_v = self.v_ + self.R_.as_matrix() @ (imu.acce_.reshape(-1, 1) -
                                                 self.ba_) * dt + self.g_ * dt
        new_R = SO3(self.R_.as_matrix() @
                    SO3.exp((imu.gyro_ - self.bg_.reshape(-1))*dt).as_matrix())
        # 更新名义状态，ba,bg,g不变
        self.P_ = new_p
        self.v_ = new_v
        self.R_ = new_R

        # 误差递推，(3.47)
        # 计算运动过程雅可比矩阵 F
        F = np.identity(18)
        F[0:3, 3:6] = np.identity(3) * dt
        F[3:6, 6:9] = -1 * self.R_.as_matrix() @ SO3.wedge(imu.acce_ -
                                                           self.ba_.reshape(-1)) * dt
        F[3:6, 12:15] = -1 * self.R_.as_matrix() * dt
        F[3:6, 15:18] = np.identity(3) * dt
        F[6:9, 6:9] = SO3.exp(
            ((imu.gyro_-self.bg_.reshape(-1))*dt*-1)).as_matrix()
        F[6:9, 9:12] = -np.identity(3) * dt

        # 更新协方差矩阵，(3.48)
        self.dx_ = F @ self.dx_
        self.cov_ = F @ self.cov_ @ F.T + self.Q_
        self.last_imu_ = imu
        return True

    def UpdateAndReset(self):
        # 名义变量+误差变量
        self.P_ = self.P_ + self.dx_[0:3, 0:1]
        self.v_ = self.v_ + self.dx_[3:6, 0:1]
        self.R_ = SO3(self.R_.as_matrix() @
                      SO3.exp(self.dx_[6:9, 0:1].reshape(-1)).as_matrix())
        if self.options_.update_bias_gyro_:
            self.bg_ = self.bg_ + self.dx_[9:12, 0:1]
        if self.options_.update_bias_acce_:
            self.ba_ = self.ba_ + self.dx_[12:15, 0:1]
        self.g_ = self.g_ + self.dx_[15:18, 0:1]

        # P阵进行投影，(3.63)
        J = np.identity(18)
        J[6:9, 6:9] = np.identity(
            3) - SO3.wedge(self.dx_[6:9, 0:1].reshape(-1))*0.5
        self.cov_ = J @ self.cov_ @ J.T

        # 清空误差变量
        self.dx_ = np.zeros(18).reshape(-1, 1)

    def ObserveWheelSpeed(self, velo: np.array, velo_noise: float) -> bool:
        """轮速计观测

        Args:
            odom (np.array): 观测平移
            odom_noise (float): 平移噪声

        Returns:
            bool: _description_
        """
        # 观测状态变量中的速度，H为6x18，其余为零
        H = np.zeros([3, 18])
        H[0:3, 3:6] = np.identity(3)

        # 卡尔曼增益和更新过程
        noise_cov = np.diag(
            [velo_noise, velo_noise, velo_noise])
        K = self.cov_ @ H.T @ np.linalg.inv(H @ self.cov_ @  H.T + noise_cov)

        # velocity obs
        average_vel = 0.5 * (velo[0] + velo[1])

        # 更新x和cov
        innov = np.zeros(3).reshape(-1, 1)
        innov[0:3, 0:1] = self.R_.as_matrix() @ np.array([average_vel, 0, 0]
                                                         ).reshape(-1, 1) - self.v_
        self.dx_ = K @ innov
        self.cov_ = (np.identity(18) - K @ H) @ self.cov_
        self.UpdateAndReset()
        return True

    def ObserveSE3(self, pose_tran: np.array, pose_SO3: np.array, trans_noise: float, ang_noise: float) -> bool:
        """SE3观测

        Args:
            pose_tran (np.array): 观测平移
            pose_SO3 (np.array): 观测平移旋转
            trans_noise (float): 平移噪声
            ang_noise (float): 旋转噪声

        Returns:
            bool: _description_
        """
        # 观测状态变量中的p, R，H为6x18，其余为零
        H = np.zeros([6, 18])
        H[0:3, 0:3] = np.identity(3)  # p
        H[3:6, 6:9] = np.identity(3)  # R

        # 卡尔曼增益和更新过程
        noise_cov = np.diag(
            [trans_noise, trans_noise, trans_noise, ang_noise, ang_noise, ang_noise])
        K = self.cov_ @ H.T @ np.linalg.inv(H @ self.cov_ @  H.T + noise_cov)

        # 更新x和cov
        innov = np.zeros(6).reshape(-1, 1)
        innov[0:3, 0:1] = pose_tran - self.P_
        innov[3:6, 0:1] = SO3(self.R_.inv().as_matrix() @
                              pose_SO3).log().reshape(-1, 1)
        self.dx_ = K @ innov
        self.cov_ = (np.identity(18) - K @ H) @ self.cov_
        self.UpdateAndReset()
        return True

    def ObserveTran(self, pose_tran: np.array, trans_noise: float) -> bool:
        """位置观测

        Args:
            pose_tran (np.array): 观测平移
            trans_noise (float): 平移噪声

        Returns:
            bool: _description_
        """
        # 观测状态变量中的p，H为3x18，其余为零
        H = np.zeros([3, 18])
        H[0:3, 0:3] = np.identity(3)  # p

        # 卡尔曼增益和更新过程
        noise_cov = np.diag([trans_noise, trans_noise, trans_noise])
        K = self.cov_ @ H.T @ np.linalg.inv(H @ self.cov_ @  H.T + noise_cov)

        # 更新x和cov
        innov = np.zeros(3).reshape(-1, 1)
        innov[0:3, 0:1] = pose_tran - self.P_

        self.dx_ = K @ innov
        self.cov_ = (np.identity(18) - K @ H) @ self.cov_
        self.UpdateAndReset()
        return True
