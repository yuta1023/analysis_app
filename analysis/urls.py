from django.urls import path
from . import views

app_name = 'analysis'
urlpatterns = [
    path('', views.menu, name='menu'),
    path('pole_ebsd', views.pole_ebsd, name="pole_ebsd"),
    path('pole_ebsd/plot', views.img_pole_ebsd, name='img_pole_ebsd'),
    path('pole_overplot', views.pole_overplot, name='pole_overplot'),
    path('pole_overplot/plot', views.img_pole_overplot, name='img_pole_overplot'),
    path('direction', views.direction_analysis, name='direction_analysis'),
    path('direction/plot', views.img_direction, name='img_direction'),
    path('plane', views.plane_analysis, name='plane_analysis'),
    path('plane/plot',views.img_plane, name='img_plane'),
    path('ks_one', views.ks_one, name='ks_one'),
    path('ks_all', views.ks_all, name='ks_all'),
]
