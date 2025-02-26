from django.contrib import admin
from .models import Metric, Analityc, Confusion_Matrix
# Register your models here.

class MetricAdmin(admin.ModelAdmin):
    readonly_fields = ('created', 'updated')
    list_display = ('epoch', 'accuracy', 'loss', 'brand', 'created')  # Opcional: mostrar campos en la lista
    search_fields = ('brand',)  # Opcional: búsqueda por marca

class AnalitycAdmin(admin.ModelAdmin):
    readonly_fields = ('created', 'updated')
    list_display = ('description', 'average', 'created')  # Opcional: mostrar campos en la lista

class Confusion_MatrixAdmin(admin.ModelAdmin):
    readonly_fields = ('created', 'updated')
    list_display = ('precision', 'recall', 'f1_score', 'support','created')  # Opcional: mostrar campos en la lista

admin.site.register(Metric, MetricAdmin)
admin.site.register(Analityc, AnalitycAdmin)
admin.site.register(Confusion_Matrix, Confusion_MatrixAdmin)