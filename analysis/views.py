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
from . import function_pole
from . import function_3d
from . import function_ks
from . import function_ipf

nd = None
rd = None
nd1 = None
rd1 = None
nd2 = None
rd2 = None
nd_dir = None
rd_dir = None
phi_dir = None
theta_dir = None
rot_dir = None
graph_dir = None
nd_pla = None
rd_pla = None
rot_pla = None
graph_pla = None
df_xyz = None


# PNG画像形式に変換
def plt2png():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    s = buf.getvalue()
    buf.close()
    return s


# Create your views here.
def menu(request):
    return render(request, 'analysis/menu.html')


def pole_ebsd(request):
    global nd
    global rd
    f1 = forms.NdForm()
    f2 = forms.RdForm()
    nd_h = request.POST.get('nd_h', 0)
    nd_k = request.POST.get('nd_k', 0)
    nd_l = request.POST.get('nd_l', 1)
    rd_h = request.POST.get('rd_h', 0)
    rd_k = request.POST.get('rd_k', -1)
    rd_l = request.POST.get('rd_l', 0)
    nd_pre = [nd_h, nd_k, nd_l]
    rd_pre = [rd_h, rd_k, rd_l]
    nd = [float(s) for s in nd_pre]
    rd = [float(s) for s in rd_pre]
    dic = {'nd_form': f1,
           'rd_form': f2,
           'nd': nd,
           'rd': rd,
           }
    return render(request, 'analysis/pole_ebsd.html', dic)


def img_pole_ebsd(request):
    global nd
    global rd
    plt.clf()
    co_list = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    aa, a_inverse = function_pole.crystal_matrix(nd, rd)
    co = function_pole.generate_co(co_list)
    co_norm = function_pole.co_norm(co)
    co_convert = function_pole.convert_inverse(co_norm, a_inverse)
    r_theta = function_pole.xyz2polar(co_convert)
    ax = function_pole.set_polar_axis()
    function_pole.polar_plot(r_theta, ax, "black")
    png = plt2png()
    plt.cla()
    response = HttpResponse(png, content_type='image/png')
    return response


def pole_overplot(request):
    global nd1
    global rd1
    global nd2
    global rd2
    f11 = forms.NdForm1()
    f12 = forms.RdForm1()
    f21 = forms.NdForm2()
    f22 = forms.RdForm2()
    nd_h1 = request.POST.get('nd_h1', 0)
    nd_k1 = request.POST.get('nd_k1', 0)
    nd_l1 = request.POST.get('nd_l1', 1)
    rd_h1 = request.POST.get('rd_h1', 0)
    rd_k1 = request.POST.get('rd_k1', -1)
    rd_l1 = request.POST.get('rd_l1', 0)
    nd_h2 = request.POST.get('nd_h2', 1)
    nd_k2 = request.POST.get('nd_k2', 2)
    nd_l2 = request.POST.get('nd_l2', 1)
    rd_h2 = request.POST.get('rd_h2', -1)
    rd_k2 = request.POST.get('rd_k2', 0)
    rd_l2 = request.POST.get('rd_l2', 1)
    nd_pre1 = [nd_h1, nd_k1, nd_l1]
    rd_pre1 = [rd_h1, rd_k1, rd_l1]
    nd_pre2 = [nd_h2, nd_k2, nd_l2]
    rd_pre2 = [rd_h2, rd_k2, rd_l2]
    nd1 = [float(s) for s in nd_pre1]
    rd1 = [float(s) for s in rd_pre1]
    nd2 = [float(s) for s in nd_pre2]
    rd2 = [float(s) for s in rd_pre2]
    dic = {'nd_form1': f11,
           'rd_form1': f12,
           'nd_form2': f21,
           'rd_form2': f22,
           'nd1': nd1,
           'rd1': rd1,
           'nd2': nd2,
           'rd2': rd2,
           }
    return render(request, 'analysis/pole_overplot.html', dic)


def img_pole_overplot(request):
    global nd1
    global rd1
    global nd2
    global rd2
    plt.clf()
    co_list = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    aa1, a_inverse1 = function_pole.crystal_matrix(nd1, rd1)
    aa2, a_inverse2 = function_pole.crystal_matrix(nd2, rd2)
    co = function_pole.generate_co(co_list)
    co_norm = function_pole.co_norm(co)
    co_convert1 = function_pole.convert_inverse(co_norm, a_inverse1)
    co_convert2 = function_pole.convert_inverse(co_norm, a_inverse2)
    r_theta1 = function_pole.xyz2polar(co_convert1)
    r_theta2 = function_pole.xyz2polar(co_convert2)
    ax = function_pole.set_polar_axis()
    function_pole.polar_plot(r_theta1, ax, "red")
    function_pole.polar_plot(r_theta2, ax, "blue")
    png = plt2png()
    plt.cla()
    response = HttpResponse(png, content_type='image/png')
    return response


def direction_analysis(request):
    global nd_dir
    global rd_dir
    global phi_dir
    global theta_dir
    global rot_dir
    global graph_dir
    f1_d = forms.NdForm()
    f2_d = forms.RdForm()
    f_p = forms.PhiForm()
    f_t = forms.ThetaForm()
    f_r = forms.RotationForm()
    nd_h = request.POST.get('nd_h', 0)
    nd_k = request.POST.get('nd_k', 0)
    nd_l = request.POST.get('nd_l', 1)
    rd_h = request.POST.get('rd_h', 0)
    rd_k = request.POST.get('rd_k', -1)
    rd_l = request.POST.get('rd_l', 0)
    phi = request.POST.get('phi', 45)
    theta = request.POST.get('theta', 45)
    rot = request.POST.get('rot', 90)
    graph_dir = request.POST.get('graph_type', 'pf')
    if graph_dir == 'pf':
        graph_type = 'Pole figure'
    elif graph_dir == 'ipf':
        graph_type = 'Inverse pole figure'
    else:
        graph_type = 'Pole figure'
    nd_pre = [nd_h, nd_k, nd_l]
    rd_pre = [rd_h, rd_k, rd_l]
    nd_dir = [float(s) for s in nd_pre]
    rd_dir = [float(s) for s in rd_pre]
    phi_dir = float(phi)
    theta_dir = float(theta)
    rot_dir = float(rot)
    dic = {'nd_form': f1_d,
           'rd_form': f2_d,
           'phi_form': f_p,
           'theta_form': f_t,
           'rot_form': f_r,
           'nd': nd_dir,
           'rd': rd_dir,
           'phi': phi_dir,
           'theta': theta_dir,
           'rot': rot_dir,
           'graph_type': graph_type,
           }
    return render(request, 'analysis/direction.html', dic)


def img_direction(request):
    global nd_dir
    global rd_dir
    global phi_dir
    global theta_dir
    global rot_dir
    global graph_dir
    plt.clf()
    phi_theta = [0, phi_dir, theta_dir]
    co_list = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 2, 5], [1, 1, 3]]
    df_dir = pd.DataFrame(np.array([phi_theta]))
    co = function_pole.generate_co(co_list)
    co_norm = function_pole.co_norm(co)
    r_theta = function_pole.xyz2polar(co_norm)
    if graph_dir == 'pf':
        a = function_3d.crystal_matrix_rot(nd_dir, rd_dir, rot_dir)
        xyz = function_3d.phi_theta2xyz(df_dir)
        cry = function_3d.xyz2co(xyz, a)
        cry_use = function_3d.south2north(cry)
        df_polar = function_3d.convert_stereo(cry_use)
        ax = function_pole.set_polar_axis()
        function_pole.zone_ax_plot(ax)
        function_pole.polar_plot(r_theta, ax, "black")
        function_3d.pol_plot(df_polar, ax, "red")
    elif graph_dir == 'ipf':
        a = function_3d.crystal_matrix_rot(nd_dir, rd_dir, rot_dir)
        xyz = function_3d.phi_theta2xyz(df_dir)
        cry = function_3d.xyz2co(xyz, a)
        cry_use = function_3d.south2north(cry)
        cry_ipf = function_ipf.change_ipf(cry_use)
        df_polar = function_3d.convert_stereo(cry_ipf)
        frame_101 = np.zeros((0, 3))
        for i in range(-10, 10):
            for s in range(-10, 10):
                x = i
                y = s
                z = i
                part = [x, y, z]
                frame_ = np.array(part)
                frame_101 = np.vstack((frame_101, frame_))
        frame_101_k = function_ipf.change_ipf(frame_101)
        frame_bottom = np.array([[1, 0, 1]])
        frame_upper = np.array([[1, 1, 1]])
        ax = function_ipf.axis_ipf2()
        pol_101 = function_ipf.co2polar(frame_101_k)
        pol_bottom = function_ipf.co2polar(frame_bottom)
        pol_bottom.loc["add"] = [0, 0, 0]
        pol_upper = function_ipf.co2polar(frame_upper)
        pol_upper.loc["add"] = [0, 0, 0]
        function_ipf.polar_line(ax, pol_101)
        function_ipf.polar_line(ax, pol_bottom)
        function_ipf.polar_line(ax, pol_upper)
        function_ipf.polar_plot_k(r_theta, ax, "black")
        function_ipf.pol_plot(df_polar, ax, "red")
    png = plt2png()
    plt.cla()
    response = HttpResponse(png, content_type='image/png')
    return response


def plane_analysis(request):
    global nd_pla
    global rd_pla
    global rot_pla
    global graph_pla
    global df_xyz
    f1_d = forms.NdForm()
    f2_d = forms.RdForm()
    f_r = forms.RotationForm()
    nd_h = request.POST.get('nd_h', 0)
    nd_k = request.POST.get('nd_k', 0)
    nd_l = request.POST.get('nd_l', 1)
    rd_h = request.POST.get('rd_h', 0)
    rd_k = request.POST.get('rd_k', -1)
    rd_l = request.POST.get('rd_l', 0)
    rot = request.POST.get('rot', 90)
    graph_pla = request.POST.get('graph_type', 'pf')
    if graph_pla == 'pf':
        graph_type = 'Pole figure'
    elif graph_pla == 'ipf':
        graph_type = 'Inverse pole figure'
    else:
        graph_type = 'Pole figure'
    nd_pre = [nd_h, nd_k, nd_l]
    rd_pre = [rd_h, rd_k, rd_l]
    nd_pla = [float(s) for s in nd_pre]
    rd_pla = [float(s) for s in rd_pre]
    rot_pla = float(rot)
    if request.method == 'POST':
        df_xyz = None
        name = request.FILES['file'].name
        ext = os.path.splitext(name)
        if ext[1] == ".csv":
            me = io.TextIOWrapper(request.FILES['file'].file, encoding='utf-8')
            df_xyz = pd.read_csv(me, header=None)
        else:
            pass
    else:
        name = None
    dic = {'nd_form': f1_d,
           'rd_form': f2_d,
           'rot_form': f_r,
           'nd': nd_pla,
           'rd': rd_pla,
           'rot': rot_pla,
           'file_name': name,
           'graph_type': graph_type,
           }
    return render(request, 'analysis/plane.html', dic)


def img_plane(request):
    global nd_pla
    global rd_pla
    global rot_pla
    global graph_pla
    global df_xyz
    if df_xyz is None:
        response = None
    else:
        plt.clf()
        co_list = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 2, 5], [1, 1, 3]]
        co = function_pole.generate_co(co_list)
        co_norm = function_pole.co_norm(co)
        r_theta = function_pole.xyz2polar(co_norm)
        if graph_pla == 'pf':
            a = function_3d.crystal_matrix_rot(nd_pla, rd_pla, rot_pla)
            xyz = np.array(df_xyz)
            cry = function_3d.xyz2co(xyz, a)
            cry_use = function_3d.south2north(cry)
            df_polar = function_3d.convert_stereo(cry_use)
            df_r_theta = df_polar.drop("theta_rad", axis=1)
            df_hist = function_3d.create_hist(df_r_theta)
            ax = function_pole.set_polar_axis()
            function_3d.plot_hist(df_hist, ax)
            function_pole.zone_ax_plot(ax)
            function_pole.polar_plot(r_theta, ax, "black")
        elif graph_pla == 'ipf':
            frame_101 = np.zeros((0, 3))
            for i in range(-15, 15):
                for s in range(-15, 15):
                    x = i
                    y = s
                    z = i
                    part = [x, y, z]
                    frame_ = np.array(part)
                    frame_101 = np.vstack((frame_101, frame_))
            frame_bottom = np.array([[1, 0, 1]])
            frame_upper = np.array([[1, 1, 1]])
            a = function_3d.crystal_matrix_rot(nd_pla, rd_pla, rot_pla)
            xyz = np.array(df_xyz)
            cry = function_3d.xyz2co(xyz, a)
            cry_use = function_3d.south2north(cry)
            cry_ipf = function_ipf.change_ipf(cry_use)
            df_polar = function_3d.convert_stereo(cry_ipf)
            df_r_theta = df_polar.drop("theta_rad", axis=1)
            df_hist = function_3d.create_hist(df_r_theta)
            ax = function_ipf.axis_ipf()
            pol_101 = function_ipf.co2polar(frame_101)
            pol_bottom = function_ipf.co2polar(frame_bottom)
            pol_bottom.loc["add"] = [0, 0, 0]
            pol_upper = function_ipf.co2polar(frame_upper)
            pol_upper.loc["add"] = [0, 0, 0]
            function_ipf.polar_line(ax, pol_101)
            function_ipf.polar_line(ax, pol_bottom)
            function_ipf.polar_line(ax, pol_upper)
            co = function_pole.generate_co(co_list)
            co_norm = function_pole.co_norm(co)
            r_theta = function_pole.xyz2polar(co_norm)
            xx = np.ones(len(pol_101))
            yy = pd.DataFrame(xx.T)
            zz = pol_101.join(yy)
            function_ipf.plot_hist_k(df_hist, ax)
            function_ipf.polar_plot_k(r_theta, ax, "black")
            ax.fill_between(zz.iloc[:, 2], zz.iloc[:, 0], zz.iloc[:, 3], facecolors="white")
        png = plt2png()
        plt.cla()
        response = HttpResponse(png, content_type='image/png')
    return response


def ks_one(request):
    f11 = forms.NdForm1()
    f12 = forms.RdForm1()
    f21 = forms.NdForm2()
    f22 = forms.RdForm2()
    p1 = forms.PlaneForm1()
    p2 = forms.PlaneForm2()
    d1 = forms.DirectionForm1()
    d2 = forms.DirectionForm2()
    nd_h1 = request.POST.get('nd_h1', -2)
    nd_k1 = request.POST.get('nd_k1', 1)
    nd_l1 = request.POST.get('nd_l1', 2)
    rd_h1 = request.POST.get('rd_h1', 1.5)
    rd_k1 = request.POST.get('rd_k1', -1)
    rd_l1 = request.POST.get('rd_l1', 0)
    nd_h2 = request.POST.get('nd_h2', 1.5)
    nd_k2 = request.POST.get('nd_k2', 0)
    nd_l2 = request.POST.get('nd_l2', 1)
    rd_h2 = request.POST.get('rd_h2', 1)
    rd_k2 = request.POST.get('rd_k2', 1)
    rd_l2 = request.POST.get('rd_l2', -1)
    p1_h = request.POST.get('p1_h', -1)
    p1_k = request.POST.get('p1_k', -1)
    p1_l = request.POST.get('p1_l', 1)
    p2_h = request.POST.get('p2_h', 1)
    p2_k = request.POST.get('p2_k', -1)
    p2_l = request.POST.get('p2_l', 0)
    d1_h = request.POST.get('d1_h', -1)
    d1_k = request.POST.get('d1_k', 1)
    d1_l = request.POST.get('d1_l', 0)
    d2_h = request.POST.get('d2_h', -1)
    d2_k = request.POST.get('d2_k', -1)
    d2_l = request.POST.get('d2_l', 1)
    nd_pre1 = [nd_h1, nd_k1, nd_l1]
    rd_pre1 = [rd_h1, rd_k1, rd_l1]
    nd_pre2 = [nd_h2, nd_k2, nd_l2]
    rd_pre2 = [rd_h2, rd_k2, rd_l2]
    p1_pre = [p1_h, p1_k, p1_l]
    p2_pre = [p2_h, p2_k, p2_l]
    d1_pre = [d1_h, d1_k, d1_l]
    d2_pre = [d2_h, d2_k, d2_l]
    nd_fcc = [float(s) for s in nd_pre1]
    rd_fcc = [float(s) for s in rd_pre1]
    nd_bcc = [float(s) for s in nd_pre2]
    rd_bcc = [float(s) for s in rd_pre2]
    p_fcc = [float(s) for s in p1_pre]
    p_bcc = [float(s) for s in p2_pre]
    d_fcc = [float(s) for s in d1_pre]
    d_bcc = [float(s) for s in d2_pre]
    x_ = p_fcc / np.linalg.norm(p_fcc)
    y_ = p_bcc / np.linalg.norm(p_bcc)
    xx_ = d_fcc / np.linalg.norm(d_fcc)
    yy_ = d_bcc / np.linalg.norm(d_bcc)
    comb_list = p_fcc + d_fcc + p_bcc + d_bcc
    norm_list = list(x_) + list(xx_) + list(y_) + list(yy_)
    comb_array = np.array([comb_list])
    norm_array = np.array([norm_list])
    result = function_ks.calc_ks(nd_fcc, rd_fcc, nd_bcc, rd_bcc, comb_array, norm_array)
    dic = {'nd_form1': f11,
           'rd_form1': f12,
           'nd_form2': f21,
           'rd_form2': f22,
           'p1_form': p1,
           'p2_form': p2,
           'd1_form': d1,
           'd2_form': d2,
           'nd_fcc': nd_fcc,
           'rd_fcc': rd_fcc,
           'nd_bcc': nd_bcc,
           'rd_bcc': rd_bcc,
           'p_fcc': p_fcc,
           'p_bcc': p_bcc,
           'd_fcc': d_fcc,
           'd_bcc': d_bcc,
           'plane': round(result[4], 2),
           'direction': round(result[5], 2),
           'ks': round(result[6], 2),
           }
    return render(request, 'analysis/ks_one.html', dic)


def ks_all(request):
    f11 = forms.NdForm1()
    f12 = forms.RdForm1()
    f21 = forms.NdForm2()
    f22 = forms.RdForm2()
    nd_h1 = request.POST.get('nd_h1', -2)
    nd_k1 = request.POST.get('nd_k1', 1)
    nd_l1 = request.POST.get('nd_l1', 2)
    rd_h1 = request.POST.get('rd_h1', 1.5)
    rd_k1 = request.POST.get('rd_k1', -1)
    rd_l1 = request.POST.get('rd_l1', 0)
    nd_h2 = request.POST.get('nd_h2', 1.5)
    nd_k2 = request.POST.get('nd_k2', 0)
    nd_l2 = request.POST.get('nd_l2', 1)
    rd_h2 = request.POST.get('rd_h2', 1)
    rd_k2 = request.POST.get('rd_k2', 1)
    rd_l2 = request.POST.get('rd_l2', -1)
    nd_pre1 = [nd_h1, nd_k1, nd_l1]
    rd_pre1 = [rd_h1, rd_k1, rd_l1]
    nd_pre2 = [nd_h2, nd_k2, nd_l2]
    rd_pre2 = [rd_h2, rd_k2, rd_l2]
    nd_fcc = [float(s) for s in nd_pre1]
    rd_fcc = [float(s) for s in rd_pre1]
    nd_bcc = [float(s) for s in nd_pre2]
    rd_bcc = [float(s) for s in rd_pre2]
    # ===============================================================================
    # preparation of possible combination of plane and direction
    # K-S OR
    # (111)fcc//(011)bcc
    # [0-11]fcc//[1-11]bcc
    # ===============================================================================
    x1 = [[1, 1, 1], [-1, -1, -1]]
    y1 = [[0, 1, -1], [0, -1, 1], [1, 0, -1], [-1, 0, 1], [1, -1, 0], [-1, 1, 0]]
    x2 = [[1, -1, 1], [-1, 1, -1]]
    y2 = [[0, 1, 1], [0, -1, -1], [1, 0, -1], [-1, 0, 1], [1, 1, 0], [-1, -1, 0]]
    x3 = [[1, 1, -1], [-1, -1, 1]]
    y3 = [[0, 1, 1], [0, -1, -1], [1, 0, 1], [-1, 0, -1], [1, -1, 0], [-1, 1, 0]]
    x4 = [[-1, 1, 1], [1, -1, -1]]
    y4 = [[0, 1, -1], [0, -1, 1], [1, 0, 1], [-1, 0, -1], [1, 1, 0], [-1, -1, 0]]

    a1 = [[0, 1, 1], [0, -1, -1]]
    b1 = [[1, 1, -1], [-1, 1, -1], [1, -1, 1], [-1, -1, 1]]
    a2 = [[1, 0, 1], [-1, 0, -1]]
    b2 = [[1, 1, -1], [1, -1, -1], [-1, 1, 1], [-1, -1, 1]]
    a3 = [[1, 1, 0], [-1, -1, 0]]
    b3 = [[1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1]]
    a4 = [[0, 1, -1], [0, -1, 1]]
    b4 = [[1, 1, 1], [-1, 1, 1], [1, -1, -1], [-1, -1, -1]]
    a5 = [[1, 0, -1], [-1, 0, 1]]
    b5 = [[1, 1, 1], [1, -1, 1], [-1, 1, -1], [-1, -1, -1]]
    a6 = [[1, -1, 0], [-1, 1, 0]]
    b6 = [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
    fcc_comb = []
    fcc_norm = []
    for s, t in zip([x1, x2, x3, x4], [y1, y2, y3, y4]):
        for v, w in itertools.product(s, t):
            z = list(v) + list(w)
            zz = list(list(v) / np.linalg.norm(list(v))) + list(list(w) / np.linalg.norm(list(w)))
            fcc_comb.append(z)
            fcc_norm.append(zz)
    bcc_comb = []
    bcc_norm = []
    for s, t in zip([a1, a2, a3, a4, a5, a6], [b1, b2, b3, b4, b5, b6]):
        for v, w in itertools.product(s, t):
            z = list(v) + list(w)
            zz = list(list(v) / np.linalg.norm(list(v))) + list(list(w) / np.linalg.norm(list(w)))
            bcc_comb.append(z)
            bcc_norm.append(zz)
    total_comb = []
    for v, w in itertools.product(fcc_comb, bcc_comb):
        z = list(v) + list(w)
        total_comb.append(z)
    comb_array = np.array(total_comb)
    norm_comb = []
    for v, w in itertools.product(fcc_norm, bcc_norm):
        z = list(v) + list(w)
        norm_comb.append(z)
    norm_array = np.array(norm_comb)
    result = function_ks.calc_ks(nd_fcc, rd_fcc, nd_bcc, rd_bcc, comb_array, norm_array)
    dic = {'nd_form1': f11,
           'rd_form1': f12,
           'nd_form2': f21,
           'rd_form2': f22,
           'nd_fcc': nd_fcc,
           'rd_fcc': rd_fcc,
           'nd_bcc': nd_bcc,
           'rd_bcc': rd_bcc,
           'p_fcc': result[0],
           'p_bcc': result[2],
           'd_fcc': result[1],
           'd_bcc': result[3],
           'plane': round(result[4], 2),
           'direction': round(result[5], 2),
           'ks': round(result[6], 2),
           }
    return render(request, 'analysis/ks_all.html', dic)
