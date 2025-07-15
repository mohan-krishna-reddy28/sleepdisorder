# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home'),  # index.html
#     path('input/', views.input, name='input'),  # input.html
#     path('output/', views.output, name='output'),  # output.html
#     path('team/', views.team, name='team'),  # team.html
#     path('about/', views.about, name='about'),  # about.html
# ]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # index.html
    path('base/',views.base,name='base'),
    path('input/', views.input, name='input'),  # input.html
    path('output/', views.output, name='output'),  # output.html
    path('team/', views.team, name='team'),  # team.html
    path('about/', views.about, name='about'),  # about.html
]
