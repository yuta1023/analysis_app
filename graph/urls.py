from django.urls import path
from . import views

app_name = 'graph'
urlpatterns = [
    path('', views.graph_menu, name='graph_menu'),
    path('2d_plot', views.plot_2d, name='plot_2d'),
    path('2d_plot/plot', views.img_2d_plot, name='img_2d_plot'),
    path('2d_hist', views.hist_2d, name='hist_2d'),
    path('2d_hist/plot', views.img_hist_2d, name='img_hist_2d'),
]
