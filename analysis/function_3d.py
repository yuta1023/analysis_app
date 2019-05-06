import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# 結晶方位行列計算(回転考慮) ex. nd=[1,1,1] rd = [2,1,2], 反時計回りr度
def crystal_matrix_rot(nd, rd, rot_deg):
    # TDを計算
    td = np.cross(nd, rd)
    nd_norm = nd / np.linalg.norm(nd)
    rd_norm = rd / np.linalg.norm(rd)
    td_norm = td / np.linalg.norm(td)
    # 結晶方位行列
    aa = np.matrix([[-td_norm[0], -rd_norm[0], -nd_norm[0]],
                    [-td_norm[1], -rd_norm[1], -nd_norm[1]],
                    [-td_norm[2], -rd_norm[2], -nd_norm[2]]])
    # 結晶方位行列の回転
    rot = np.deg2rad(rot_deg)
    r_matrix = np.matrix([[np.cos(rot), -np.sin(rot), 0],
                          [np.sin(rot), np.cos(rot), 0],
                          [0, 0, 1]])
    a = np.dot(aa, r_matrix)
    return a


# 極座標から直交座標変換 df[phi, theta]
def phi_theta2xyz(df_dir):
    dir_deg = np.array(df_dir.iloc[:, 1:3])
    dir_rad = np.radians(dir_deg)
    xyz = []
    for i in range(0, dir_rad.shape[0]):
        xyz_ = [np.sin(dir_rad[i, 0]) * np.cos(dir_rad[i, 1]), np.sin(dir_rad[i, 0]) * np.sin(dir_rad[i, 1]),
                np.cos(dir_rad[i, 0])]
        xyz.append(xyz_)
    xyz = np.array(xyz)
    return xyz


# 直交座標から化粧方位座標への変換
def xyz2co(xyz, a):
    crystal = []
    for i in range(0, xyz.shape[0]):
        xyzt = np.array(xyz[i, :])
        crystal_ = np.dot(a, xyzt)
        crystal_ = [crystal_[0, 0], crystal_[0, 1], crystal_[0, 2]]
        crystal.append(crystal_)
    crystal = np.array(crystal)
    return crystal


# 南半球を北半球に変換
def south2north(crystal):
    crystal_n = []
    for i in range(0, crystal.shape[0]):
        if crystal[i, 2] < 0:
            xc_ = -crystal[i, 0]
            yc_ = -crystal[i, 1]
            zc_ = -crystal[i, 2]
        else:
            xc_ = crystal[i, 0]
            yc_ = crystal[i, 1]
            zc_ = crystal[i, 2]
        crystal_n_ = [xc_, yc_, zc_]
        crystal_n.append(crystal_n_)
    crystal_n = np.array(crystal_n)
    return crystal_n


# ステレオ投影座標に変換
def convert_stereo(crystal_n):
    results = []
    for i in range(0, crystal_n.shape[0]):
        xx = crystal_n[i, 0]
        yy = crystal_n[i, 1]
        zz = crystal_n[i, 2]
        r = np.sqrt((1 - zz) / (1 + zz))
        theta = np.arctan(yy / xx) * (180 / np.pi) + ((xx - np.abs(xx)) / (2 * xx)) * 180
        theta_rad = theta / 180 * np.pi
        sets = [r, theta, theta_rad]
        results.append(sets)
    results = np.array(results)
    df_polar = pd.DataFrame(results, columns=["r", "theta", "theta_rad"])
    return df_polar


# グラフ準備はfunction_pole.pyからすること


# 極座標プロット
def pol_plot(r_theta, ax, color):
    t = r_theta.iloc[:, 2]
    rr = r_theta.iloc[:, 0]
    ax.scatter(t, rr, c=color, s=30)


# ヒストグラム作成
def create_hist(df_r_thrta):
    # ヒストグラム作成
    r_bin = np.arange(-0.1, 1.01, 0.1)
    theta_bin = np.arange(-90, 271, 10)
    r_list = df_r_thrta.iloc[:, 0].values
    theta_list = df_r_thrta.iloc[:, 1].values
    # 2次元ヒストグラム作成
    h, r_bin, theta_bin = np.histogram2d(theta_list, r_list, bins=(theta_bin, r_bin))
    # r→ビン終端, theta→ビン中心
    r_bin_fin = np.arange(0, 1.01, 0.1)
    theta_bin_cen = np.arange(-85, 266, 10)
    # データフレーム化
    df_hist = pd.DataFrame(h, columns=r_bin_fin, index=theta_bin_cen)
    # theta,r,countを縦方向に並べる
    frame = None
    for i in range(0, len(theta_bin_cen)):
        theta = np.full(11, theta_bin_cen[i])
        r = r_bin_fin
        count = df_hist.iloc[i, :].values
        theta_arr = np.array([theta])
        r_arr = np.array([r])
        count_arr = np.array([count])
        if i == 0:
            frame = np.hstack((theta_arr.T, r_arr.T, count_arr.T))
        else:
            block = np.hstack((theta_arr.T, r_arr.T, count_arr.T))
            frame = np.vstack((frame, block))
    df_trc = pd.DataFrame(frame, columns=["theta", "r", "count"])
    # カウントの総和
    count_sum = np.sum(df_trc.iloc[:, 2].values)
    # r=0.1のときの頻度を計算
    freq_01 = []
    for i in range(0, 36):
        num = 1 + 11 * i
        freq = df_trc.iat[num, 2] / ((np.cos(2 * np.arctan(df_trc.iat[num, 1] - 0.1)) - np.cos(
            2 * np.arctan(df_trc.iat[num, 1]))) / 36 * count_sum)
        freq_01.append(freq)
    freq_0 = np.mean(freq_01)
    # 頻度値を計算する
    freq_all = []
    for i in range(0, len(df_trc)):
        if i % 11 == 0:
            freq = freq_0
        else:
            freq = df_trc.iat[i, 2] / ((np.cos(2 * np.arctan(df_trc.iat[i, 1] - 0.1)) - np.cos(
                2 * np.arctan(df_trc.iat[i, 1]))) / 36 * count_sum)
        freq_all.append(freq)
    # データフレームをr,thetaの格子状のデータフレームに
    zz = None
    for i in range(0, 36):
        num = 11 * i
        xx = freq_all[num:num + 11]
        if i == 0:
            zz = np.array([xx])
        else:
            yy = np.array([xx])
            zz = np.vstack((zz, yy))
    df_for_polar = pd.DataFrame(zz, columns=r_bin_fin, index=theta_bin_cen)
    # グラフをきれいにするためにtheta_bin_cen=275(=-85)を追加
    plus = np.array([df_for_polar.iloc[0, :].values])
    df_plus = pd.DataFrame(plus, columns=r_bin_fin, index=[275])
    df_for_graph = pd.concat([df_for_polar, df_plus])
    return df_for_graph


def plot_hist(df_for_graph, ax):
    r_bin_fin = np.arange(0, 1.01, 0.1)
    theta_bin_cen_ex = np.arange(-85, 276, 10)
    data = np.array(df_for_graph)
    ctf = ax.contourf(theta_bin_cen_ex * np.pi / 180, r_bin_fin, data.T, 150, cmap=cm.jet)
    plt.colorbar(ctf, pad=0.15, orientation="vertical")
