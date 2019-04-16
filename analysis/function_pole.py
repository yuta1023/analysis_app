import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt


# 結晶方位行列計算 ex. nd=[1,1,1] rd = [2,1,2]
def crystal_matrix(nd, rd):
    # TDを計算
    td = np.cross(nd, rd)
    nd_norm = nd / np.linalg.norm(nd)
    rd_norm = rd / np.linalg.norm(rd)
    td_norm = td / np.linalg.norm(td)
    # 結晶方位行列
    a = np.matrix([[td_norm[0], -rd_norm[0], nd_norm[0]],
                   [td_norm[1], -rd_norm[1], nd_norm[1]],
                   [td_norm[2], -rd_norm[2], nd_norm[2]]])
    # 結晶方位行列の逆行列
    a_inverse = np.linalg.inv(a)
    return a, a_inverse


# 指数データの作成 ex. co_list = [[011],[111],....]
def generate_co(co_list):
    frame = np.zeros((0, 4))
    for t in range(0, len(co_list)):
        x = co_list[t][0]
        y = co_list[t][1]
        z = co_list[t][2]
        frame_co = np.zeros((0, 3))
        frame_n = np.zeros((0, 1))
        for u, v, w in itertools.product([x, -x], [y, -y], [z, -z]):
            for i in itertools.permutations([u, v, w]):
                r = np.array(i)
                rr = [int(s) for s in r]
                name = np.array([str(r[0]) + str(r[1]) + str(r[2])])
                frame_co = np.vstack([frame_co, rr])
                frame_n = np.vstack([frame_n, name])
        frame_z = np.hstack([frame_co, frame_n])
        frame = np.vstack([frame, frame_z])
    df_co = pd.DataFrame(frame, columns=["x", "y", "z", "name"]).drop_duplicates().reset_index(drop=True)
    return df_co


# 指数データフレームの正規化 出力:配列とデータフレーム
def co_norm(df):
    norm_array = []
    name = df["name"]
    for i in range(0, len(df)):
        vec = df.iloc[i, 0:3].values
        vec = [float(s) for s in vec]
        vec_norm = vec / np.linalg.norm(vec)
        norm_array.append(vec_norm)
    norm_df = pd.concat([pd.DataFrame(np.array(norm_array), columns=["x", "y", "z"]), name], axis=1)
    return norm_df


# 軸変換(001-EBSD)しプロットに使用するもののみ抽出
def convert_inverse(df, a_inverse):
    name = df["name"]
    co_array = df.iloc[:, 0:3].values
    co_ = np.dot(a_inverse, np.array(co_array).T)
    co_a = pd.concat([pd.DataFrame(co_.T, columns=["x", "y", "z"]), name], axis=1)
    co_a_use = co_a[co_a["z"] >= 0].reset_index(drop=True)
    return co_a_use


# xyz-rtheta変換
def xyz2polar(df):
    name_use = df["name"]
    r_theta_frame = []
    for i in range(0, len(df)):
        xx = df.iloc[i, 0]
        yy = df.iloc[i, 1]
        zz = df.iloc[i, 2]
        r = np.sqrt((1 - zz) / (1 + zz))
        if xx == 0:
            if yy >= 0:
                theta = 90
            else:
                theta = -90
        else:
            theta = np.arctan(yy / xx) * (180 / np.pi) + ((xx - np.abs(xx)) / (2 * xx)) * 180
        theta_rad = np.deg2rad(theta)
        r_set = [r, theta, theta_rad]
        r_theta_frame.append(r_set)
    r_theta = pd.concat([pd.DataFrame(np.array(r_theta_frame), columns=["r", "theta", "theta(rad)"]), name_use], axis=1)
    return r_theta


# グラフ準備
def set_polar_axis():
    ax = plt.subplot(projection="polar")
    ax.set_rlim([0, 1])
    ax.set_rticks([0, 1])
    ax.set_yticklabels([])
    tickpos = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    plt.xticks(tickpos, ["", "", "", ""])
    return ax


# 極座標プロット colorは文字列
def polar_plot(r_theta, ax, color):
    t = r_theta.iloc[:, 2]
    rr = r_theta.iloc[:, 0]
    ax.scatter(t, rr, c=color, s=10)
    for k, v in r_theta.iterrows():
        ax.annotate(v[3], xy=(v[2], v[0]), size=7, color=color)


# 001極点図の晶帯軸プロット
def zone_ax_plot(ax):
    def co2polar(arr):
        # 正規化
        co_frame = []
        for w in range(0, len(arr)):
            vec = arr[w]
            vec_norm = vec / np.linalg.norm(vec)
            co_frame.append(vec_norm)

        co_norms = pd.concat([pd.DataFrame(np.array(co_frame), columns=["x", "y", "z"])], axis=1)
        co_use = co_norms[co_norms["z"] >= 0].reset_index(drop=True)
        r_theta_frame = []
        for ww in range(0, len(co_use)):
            xx = co_use.iloc[ww, 0]
            yy = co_use.iloc[ww, 1]
            zz = co_use.iloc[ww, 2]
            r = np.sqrt((1 - zz) / (1 + zz))
            if xx == 0:
                if yy >= 0:
                    theta = 90
                else:
                    theta = -90
            else:
                theta = np.arctan(yy / xx) * (180 / np.pi) + ((xx - np.abs(xx)) / (2 * xx)) * 180
            theta_rad = np.deg2rad(theta)
            r_set = [r, theta, theta_rad]
            r_theta_frame.append(r_set)
        r_theta = pd.concat([pd.DataFrame(np.array(r_theta_frame), columns=["r", "theta", "theta(rad)"])], axis=1)
        xxx = r_theta.sort_values(by="theta")
        return xxx

    def polar_line(pols):
        t = pols.iloc[:, 2]
        rr = pols.iloc[:, 0]
        ax.plot(t, rr, color="gray", lw=0.8)

    # -101晶帯軸
    frame_101 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = i
            y = s
            z = i
            part = [x, y, z]
            frame_ = np.array(part)
            frame_101 = np.vstack((frame_101, frame_))
    # 101晶帯軸
    frame101 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = -i
            y = s
            z = i
            part = [x, y, z]
            frame_ = np.array(part)
            frame101 = np.vstack((frame101, frame_))
    # 0-11晶帯軸
    frame0_11 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = s
            y = i
            z = i
            part = [x, y, z]
            frame_ = np.array(part)
            frame0_11 = np.vstack((frame0_11, frame_))
    # 011晶帯軸
    frame011 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = s
            y = -i
            z = i
            part = [x, y, z]
            frame_ = np.array(part)
            frame011 = np.vstack((frame011, frame_))
    # 111晶帯軸
    frame111 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = -s - i
            y = i
            z = s
            part = [x, y, z]
            frame_ = np.array(part)
            frame111 = np.vstack((frame111, frame_))
    # -111晶帯軸
    frame_111 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = s + i
            y = i
            z = s
            part = [x, y, z]
            frame_ = np.array(part)
            frame_111 = np.vstack((frame_111, frame_))
    # -1-11晶帯軸
    frame_1_11 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = s - i
            y = i
            z = s
            part = [x, y, z]
            frame_ = np.array(part)
            frame_1_11 = np.vstack((frame_1_11, frame_))
    # 1-11晶帯軸
    frame1_11 = np.zeros((0, 3))
    for i in range(-15, 15):
        for s in range(-15, 15):
            x = -s + i
            y = i
            z = s
            part = [x, y, z]
            frame_ = np.array(part)
            frame1_11 = np.vstack((frame1_11, frame_))

    # それぞれr, thetaを計算
    p = np.sqrt((1 - 1 / np.sqrt(2)) / (1 + 1 / np.sqrt(2)))
    pol_101 = co2polar(frame_101)
    pol101 = co2polar(frame101)
    pol101.loc["add"] = [1, 270, np.deg2rad(270)]
    pol0_11 = co2polar(frame0_11)
    pol011 = co2polar(frame011)
    pol011.loc["add"] = [p, 270, np.deg2rad(270)]
    pol111 = co2polar(frame111)
    pol111.loc["add"] = [p, 270, np.deg2rad(270)]
    pol_111 = co2polar(frame_111)
    pol_111.loc["add"] = [p, 270, np.deg2rad(270)]
    pol_1_11 = co2polar(frame_1_11)
    pol1_11 = co2polar(frame1_11)
    # プロット
    polar_line(pol_101)
    polar_line(pol101)
    polar_line(pol0_11)
    polar_line(pol011)
    polar_line(pol111)
    polar_line(pol_111)
    polar_line(pol_1_11)
    polar_line(pol1_11)
