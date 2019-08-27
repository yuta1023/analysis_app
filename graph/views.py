from django.http import HttpResponse
from django.shortcuts import render
from . import forms
import io
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from . import function_graph

df_plot = None
data_plot = None
plot_type = None
ax_op_plot = None

df_hist = None
num_hist = None
data_hist = None
col_hist = None
ax_op_hist = None


# PNG画像形式に変換
def plt2png():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    s = buf.getvalue()
    buf.close()
    return s


# Create your views here.
def graph_menu(request):
    return render(request, 'graph/graph_menu.html')


def plot_2d(request):
    global df_plot
    global data_plot
    global plot_type
    global ax_op_plot
    if request.method == 'POST':
        name = request.FILES['file'].name
        ext = os.path.splitext(name)
        x_num_ = request.POST.get("x_num")
        y_num_ = request.POST.get("y_num")
        x_num = int(x_num_)
        y_num = int(y_num_)
        if ext[1] == ".csv":
            me = io.TextIOWrapper(request.FILES['file'].file, encoding='utf-8')
            df_plot = pd.read_csv(me)
            x = df_plot.iloc[:, x_num - 1].values
            y = df_plot.iloc[:, y_num - 1].values
            data_plot = [x, y]
        else:
            df_plot = None
            data_plot = None
        plot_type = request.POST.get("g_type")
        set_op_plot = request.POST.get("option")
        label_x_plot = request.POST.get("label_x")
        label_y_plot = request.POST.get("label_y")
        x_min_plot_ = request.POST.get("x_min")
        x_max_plot_ = request.POST.get("x_max")
        y_min_plot_ = request.POST.get("y_min")
        y_max_plot_ = request.POST.get("y_max")
        x_min_plot = int(x_min_plot_)
        x_max_plot = int(x_max_plot_)
        y_min_plot = int(y_min_plot_)
        y_max_plot = int(y_max_plot_)
        mesh = request.POST.get("mesh")
        if set_op_plot == "custom":
            label_x_plot_ = label_x_plot
            label_y_plot_ = label_y_plot
        else:
            label_x_plot_ = "x"
            label_y_plot_ = "y"
        ax_op_plot = [set_op_plot, label_x_plot_, label_y_plot_, x_min_plot, x_max_plot, y_min_plot, y_max_plot, mesh]

    return render(request, 'graph/2d_plot.html')


def img_2d_plot(request):
    global df_plot
    global data_plot
    global plot_type
    global ax_op_plot
    if df_plot is None:
        response = HttpResponse('')
    else:
        plt.clf()
        ax = function_graph.set_ax_2d(ax_op_plot)
        function_graph.plot_2d(ax, data_plot[0], data_plot[1], plot_type)
        png = plt2png()
        plt.cla()
        response = HttpResponse(png, content_type='image/png')
    return response


def hist_2d(request):
    global df_hist
    global num_hist
    global data_hist
    global col_hist
    global ax_op_hist
    if request.method == 'POST':
        num_hist = request.POST.get("num_hist")
        first_num = request.POST.get("first_num")
        second_num = request.POST.get("second_num")
        third_num = request.POST.get("third_num")
        data_col1 = int(first_num)
        data_col2 = int(second_num)
        data_col3 = int(third_num)
        name = request.FILES['file'].name
        ext = os.path.splitext(name)
        if ext[1] == ".csv":
            me = io.TextIOWrapper(request.FILES['file'].file, encoding='utf-8')
            df_hist = pd.read_csv(me)
            x = df_hist.iloc[:, data_col1 - 1][~np.isnan(df_hist.iloc[:, data_col1 - 1])]
            y = df_hist.iloc[:, data_col2 - 1][~np.isnan(df_hist.iloc[:, data_col2 - 1])]
            z = df_hist.iloc[:, data_col3 - 1][~np.isnan(df_hist.iloc[:, data_col3 - 1])]
            data_hist = [x, y, z]
            col_hist = [df_hist.columns[data_col1 - 1], df_hist.columns[data_col2 - 1], df_hist.columns[data_col3 - 1]]
        else:
            df_hist = None
            data_hist = None
            col_hist = None
        set_op_hist = request.POST.get("option")
        label_x_hist = request.POST.get("label_x")
        label_y_hist = request.POST.get("label_y")
        bin_ = request.POST.get("num_bin")
        bin_hist = int(bin_)
        min_ = request.POST.get("num_min")
        max_ = request.POST.get("num_max")
        min_hist = float(min_)
        max_hist = float(max_)
        if set_op_hist == "custom":
            label_x_hist_ = label_x_hist
            label_y_hist_ = label_y_hist
        else:
            label_x_hist_ = "x"
            label_y_hist_ = "frequency"
        ax_op_hist = [set_op_hist, label_x_hist_, label_y_hist_, bin_hist, min_hist, max_hist]
    return render(request, 'graph/2d_hist.html')


def img_hist_2d(request):
    global df_hist
    global num_hist
    global data_hist
    global col_hist
    global ax_op_hist
    if df_hist is None:
        response = HttpResponse('')
    else:
        plt.clf()
        ax = function_graph.set_ax_hist(ax_op_hist)
        function_graph.hist_2d(ax, data_hist, col_hist, ax_op_hist, num_hist)
        png = plt2png()
        plt.cla()
        response = HttpResponse(png, content_type='image/png')
    return response
