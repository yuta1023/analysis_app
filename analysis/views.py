from django.http import HttpResponse
from django.shortcuts import render
from . import forms
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from . import function_pole
from . import function_3d


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
    nd_h = request.GET.get('nd_h', 0)
    nd_k = request.GET.get('nd_k', 0)
    nd_l = request.GET.get('nd_l', 1)
    rd_h = request.GET.get('rd_h', 0)
    rd_k = request.GET.get('rd_k', -1)
    rd_l = request.GET.get('rd_l', 0)
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
    nd_h1 = request.GET.get('nd_h1', 0)
    nd_k1 = request.GET.get('nd_k1', 0)
    nd_l1 = request.GET.get('nd_l1', 1)
    rd_h1 = request.GET.get('rd_h1', 0)
    rd_k1 = request.GET.get('rd_k1', -1)
    rd_l1 = request.GET.get('rd_l1', 0)
    nd_h2 = request.GET.get('nd_h2', 1)
    nd_k2 = request.GET.get('nd_k2', 2)
    nd_l2 = request.GET.get('nd_l2', 1)
    rd_h2 = request.GET.get('rd_h2', -1)
    rd_k2 = request.GET.get('rd_k2', 0)
    rd_l2 = request.GET.get('rd_l2', 1)
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
           'rd1': rd2,
           'nd2': nd2,
           'rd2': rd2,
           }
    return render(request, 'analysis/pole_overplot.html', dic)


def img_pole_overplot(request):
    global nd1
    global rd1
    global nd2
    global rd2
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
    f1_d = forms.NdForm()
    f2_d = forms.RdForm()
    f_p = forms.PhiForm()
    f_t = forms.ThetaForm()
    f_r = forms.RotationForm()
    nd_h = request.GET.get('nd_h', 0)
    nd_k = request.GET.get('nd_k', 0)
    nd_l = request.GET.get('nd_l', 1)
    rd_h = request.GET.get('rd_h', 0)
    rd_k = request.GET.get('rd_k', -1)
    rd_l = request.GET.get('rd_l', 0)
    phi = request.GET.get('phi', 45)
    theta = request.GET.get('theta', 45)
    rot = request.GET.get('rot', 90)
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
           }
    return render(request, 'analysis/direction.html', dic)


def img_direction(request):
    global nd_dir
    global rd_dir
    global phi_dir
    global theta_dir
    global rot_dir
    phi_theta = [0, phi_dir, theta_dir]
    co_list = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 2, 5], [1, 1, 3]]
    df_dir = pd.DataFrame(np.array([phi_theta]))
    co = function_pole.generate_co(co_list)
    co_norm = function_pole.co_norm(co)
    r_theta = function_pole.xyz2polar(co_norm)
    a = function_3d.crystal_matrix_rot(nd_dir, rd_dir, rot_dir)
    xyz = function_3d.phi_theta2xyz(df_dir)
    cry = function_3d.xyz2co(xyz, a)
    cry_use = function_3d.south2north(cry)
    df_polar = function_3d.convert_stereo(cry_use)
    ax = function_pole.set_polar_axis()
    function_pole.zone_ax_plot(ax)
    function_pole.polar_plot(r_theta, ax, "black")
    function_3d.pol_plot(df_polar, ax, "red")
    png = plt2png()
    plt.cla()
    response = HttpResponse(png, content_type='image/png')
    return response


def plane_analysis(request):
    dic = {}
    return render(request, 'analysis/plane.html', dic)
