from django.shortcuts import render, get_object_or_404
from django.utils import timezone
from .models import Post


# Create your views here.

def home(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('-published_date')[:5]
    return render(request, 'info/home.html', {'posts': posts})


def info_list(request):
    posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('-published_date')
    return render(request, 'info/info_list.html', {'posts': posts})


def info_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return render(request, 'info/info_detail.html', {'post': post})


def about(request):
    return render(request, 'info/about.html')
