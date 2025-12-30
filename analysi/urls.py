from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path("", views.user_login, name="login"),
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),
    path('upload', views.upload_file, name='upload_file'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('delete/<str:filename>/', views.delete_file, name='delete_file'),
    path('delete/', views.delete_page, name='delete_page'),
    path('algorithms/', views.algorithms, name='algorithms'),
    path('raw-data/', views.raw_data_view, name='raw_data'),
    path('future/', views.future_predictions, name='future_predictions'),
    path('charts/', views.charts_page, name='charts_page'), 
    path('profit/', views.profit_analysis, name='profit_analysis'), 
    path("future_box_score/", views.future_box_score, name="future_box_score"),
    path("create-admin/", views.create_admin_once,name="create_admin_once"),

  
    
]