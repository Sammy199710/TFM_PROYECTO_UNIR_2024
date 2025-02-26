from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('SigatokaTrainingCNN/', views.metrics_view_cnn, name="SigatokaTrainingCNN"),
    path('SigatokaTrainingViT/', views.metrics_view_vit, name="SigatokaTrainingViT"),
    path('SigatokaComparison/', views.metrics_view_comparison, name="SigatokaComparison"),
]