import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_ax_2d(ax_option):
    plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
    plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in
    plt.rcParams['axes.linewidth'] = 1.0  # axis line width
    if ax_option[0] == "custom":
        if ax_option[7] == "on":
            plt.rcParams['axes.grid'] = True
        else:
            plt.rcParams['axes.grid'] = False
        ax = plt.subplot()
        ax.set_xlim([ax_option[3], ax_option[4]])
        ax.set_ylim([ax_option[5], ax_option[6]])
    else:
        ax = plt.subplot()
    ax.set_xlabel(ax_option[1])
    ax.set_ylabel(ax_option[2])
    return ax


def plot_2d(ax, x, y, type_):
    if type_ == "line":
        ax.plot(x, y)
    elif type_ == "scatter":
        ax.scatter(x, y)
    elif type_ == "both":
        ax.plot(x, y, marker="o")


def set_ax_hist(ax_option):
    plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
    plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in
    plt.rcParams['axes.linewidth'] = 1.0  # axis line width
    ax = plt.subplot()
    ax.set_xlabel(ax_option[1])
    ax.set_ylabel(ax_option[2])
    return ax


def hist_2d(ax, data, col, ax_option, num_hist):
    if num_hist == "1":
        data_use = [data[0]]
        col_use = [col[0]]
        if ax_option[0] == "custom":
            ax.hist(data_use, bins=ax_option[3], range=(ax_option[4], ax_option[5]), ec="black",
                    rwidth=0.8, label=col_use)
            plt.xticks(np.linspace(ax_option[4], ax_option[5], ax_option[3]+1))
        else:
            ax.hist(data_use, ec="black", rwidth=0.8, label=col_use)
    else:
        if num_hist == "2":
            data_use = [data[0], data[1]]
            col_use = [col[0], col[1]]
        else:
            data_use = [data[0], data[1], data[2]]
            col_use = [col[0], col[1], col[2]]
        if ax_option[0] == "custom":
            ax.hist(data_use, bins=ax_option[3], range=(ax_option[4], ax_option[5]), ec="black", label=col_use)
            plt.xticks(np.linspace(ax_option[4], ax_option[5], ax_option[3]+1))
        else:
            ax.hist(data_use, ec="black", label=col_use)
    plt.legend()

