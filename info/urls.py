from django.urls import path
from . import views

app_name = 'info'
urlpatterns = [
    path('', views.home, name='home'),
    path('info', views.info_list, name='info_list'),
    path('info/<int:pk>/', views.info_detail, name='info_detail'),
    path('about', views.about, name='about'),
    path('contact', views.contact, name='contact'),
]
