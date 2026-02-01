from django.contrib import admin
from django.urls import path
from mainsite import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.index, name="index"),
    path("analyze/", views.analyze, name="analyze"),
    path("result/", views.result, name="result"),
    path("suggest/",views.suggest_report,name="suggest_report"),
]
