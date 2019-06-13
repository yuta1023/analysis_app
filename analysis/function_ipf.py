import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def axis_ipf():
    ax_ = plt.subplot(projection="polar")
    ax_.set_rlim([0, 0.58])
    ax_.set_rticks([0, 0.58])
    ax_.set_yticklabels([])
    ax_.set_xticklabels([])
    ax_.set_thetamin(0)
    ax_.set_thetamax(45)
    ax_.set_axis_off()
    return ax_


def axis_ipf2():
    ax_ = plt.subplot(projection="polar")
    ax_.set_rlim([0, 0.58])
    ax_.set_rticks([0, 0.58])
    ax_.set_yticklabels([])
    ax_.set_xticklabels([])
    ax_.set_thetamin(-10)
    ax_.set_thetamax(50)
    plt.xticks([-10 / 180 * np.pi, 45 / 180 * np.pi], ["", ""])
    ax_.set_axis_off()
    return ax_


def change_ipf(cry_use_):
    cry_ipf_list = []
    for i in range(0, cry_use_.shape[0]):
        part_ = abs(cry_use_[i, :])
        cry_abs = np.sort(part_)
        sets = np.array([cry_abs[1], cry_abs[0], cry_abs[2]])
        cry_ipf_list.append(sets)
    cry_ipf_ = np.array(cry_ipf_list)
    return cry_ipf_


def co2polar(arr):
    co_frame = []
    for w in range(0, len(arr)):
        vec = arr[w]
        vec_norm = vec / np.linalg.norm(vec)
        co_frame.append(vec_norm)

    co_norms = pd.concat([pd.DataFrame(np.array(co_frame), columns=["x", "y", "z"])], axis=1)
    co_use = co_norms[co_norms["z"] >= 0].reset_index(drop=True)
    r_theta_frame = []
    for ww in range(0, len(co_use)):
        xx_ = co_use.iloc[ww, 0]
        yy_ = co_use.iloc[ww, 1]
        zz_ = co_use.iloc[ww, 2]
        r = np.sqrt((1 - zz_) / (1 + zz_))
        if xx_ == 0:
            if yy_ >= 0:
                theta = 90
            else:
                theta = -90
        else:
            theta = np.arctan(yy_ / xx_) * (180 / np.pi) + ((xx_ - np.abs(xx_)) / (2 * xx_)) * 180
        theta_rad = np.deg2rad(theta)
        r_set = [r, theta, theta_rad]
        r_theta_frame.append(r_set)
    r_theta_ = pd.concat([pd.DataFrame(np.array(r_theta_frame), columns=["r", "theta", "theta(rad)"])], axis=1)
    xxx = r_theta_.sort_values(by="theta")
    return xxx


def polar_line(ax_, pols):
    t = pols.iloc[:, 2]
    rr = pols.iloc[:, 0]
    ax_.plot(t, rr, color="black", lw=0.8)


def pol_plot(r_theta, ax, color):
    t = r_theta.iloc[:, 2]
    rr = r_theta.iloc[:, 0]
    ax.scatter(t, rr, c=color, s=100)


def polar_plot_k(r_theta_, ax_, color):
    t = r_theta_.iloc[:, 2]
    rr = r_theta_.iloc[:, 0]
    ax_.scatter(t, rr, c=color, s=30)
    for k, v in r_theta_.iterrows():
        ax_.annotate(v[3], xy=(v[2], v[0]), size=15, color=color)


def plot_hist_k(df_for_graph, ax_):
    r_bin_fin = np.arange(0, 1.01, 0.1)
    theta_bin_cen_ex = np.arange(-85, 276, 10)
    data = np.array(df_for_graph)
    ctf = ax_.contourf(theta_bin_cen_ex * np.pi / 180, r_bin_fin, data.T, 150, cmap="Reds")
    cbr = plt.colorbar(ctf, pad=0.15, orientation="vertical")
    cbr.set_ticks([data.min(), data.max()])
    cbr.set_ticklabels(["Low", "High"])
