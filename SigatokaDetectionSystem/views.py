import json
from django.shortcuts import render
from .models import *
from django.db.models import Q
# Create your views here.

def home(request):
   return render(request, "Home/home.html")

def metrics_view_cnn(request):
    strBrand = 'CNN'
    # Filtrar directamente solo los folds del 1 al 5
    metrics = Metric.objects.filter(
        epoch_brand=5, 
        brand=strBrand,
        folds__isnull=False,
        folds__gte=1,  # Mayor o igual a 1
        folds__lte=5   # Menor o igual a 5
    ).order_by('folds', 'epoch')
    
    # Obtener los folds únicos usando un filtro más directo
    folds = list(metrics.values_list('folds', flat=True).distinct().order_by('folds'))
    analitics = Analityc.objects.filter(brand=strBrand).first()
    
    print(f"analitics encontradas : {analitics}")
    print(f"Folds encontrados (1-5): {folds}")
    
    # Preparar datos para la plantilla
    fold_data = {}
    
    for fold in folds:
        # Obtener datos para este fold específico
        fold_metrics = metrics.filter(folds=fold)
        
        # Convertir a listas
        epochs = list(fold_metrics.values_list('epoch', flat=True))
        accuracy = list(fold_metrics.values_list('accuracy', flat=True))
        loss = list(fold_metrics.values_list('loss', flat=True))
        
        # Verificar que hay datos
        if epochs and accuracy and loss:
            # Guardar en el diccionario
            fold_data[str(fold)] = {
                'epochs': epochs,
                'accuracy': accuracy,
                'loss': loss
            }
            print(f"✅ Datos para fold {fold}: {len(epochs)} puntos")
        else:
            print(f"❌ No hay datos para fold {fold}")
    
    # Convertir a JSON para la plantilla
    context = {
        'folds': folds,
        'fold_data_json': json.dumps(fold_data),
        'title': f'Métricas {strBrand} - Folds 1-5',
        'analitics':analitics,
        'brand':strBrand
    }
    
    return render(request, 'SigatokaDetectionSystem/SigatokaDetectionSystem.html', context)

def metrics_view_vit(request):
    strBrand = 'ViT'
    # Filtrar directamente solo los folds del 1 al 5
    metrics = Metric.objects.filter(
        epoch_brand=5, 
        brand=strBrand,
        folds__isnull=False,
        folds__gte=1,  # Mayor o igual a 1
        folds__lte=5   # Menor o igual a 5
    ).order_by('folds', 'epoch')
    
    analitics = Analityc.objects.filter(brand=strBrand).first()

    # Obtener los folds únicos usando un filtro más directo
    folds = list(metrics.values_list('folds', flat=True).distinct().order_by('folds'))
    print(f"analitics encontradas : {analitics}")
    
    print(f"Folds encontrados (1-5): {folds}")
    
    # Preparar datos para la plantilla
    fold_data = {}
    
    for fold in folds:
        # Obtener datos para este fold específico
        fold_metrics = metrics.filter(folds=fold)
        
        # Convertir a listas
        epochs = list(fold_metrics.values_list('epoch', flat=True))
        accuracy = list(fold_metrics.values_list('accuracy', flat=True))
        loss = list(fold_metrics.values_list('loss', flat=True))
        
        # Verificar que hay datos
        if epochs and accuracy and loss:
            # Guardar en el diccionario
            fold_data[str(fold)] = {
                'epochs': epochs,
                'accuracy': accuracy,
                'loss': loss
            }
            print(f"✅ Datos para fold {fold}: {len(epochs)} puntos")
        else:
            print(f"❌ No hay datos para fold {fold}")
    
    # Convertir a JSON para la plantilla
    context = {
        'folds': folds,
        'fold_data_json': json.dumps(fold_data),
        'title': f'Métricas {strBrand} - Folds 1-5',
        'analitics': analitics,
        'brand':strBrand
    }
    
    return render(request, 'SigatokaDetectionSystem/SigatokaDetectionSystem.html', context)


def metrics_view_comparison(request):
    # Filtrar los datos para ViT y CNN
    metrics_vit = Metric.objects.filter(brand='ViT', epoch_brand__isnull=False).order_by('epoch')
    metrics_cnn = Metric.objects.filter(brand='CNN', epoch_brand__isnull=False).order_by('epoch')

    # Extraer listas para cada modelo
    vit_data = {
        'epochs': list(metrics_vit.values_list('epoch', flat=True)),
        'accuracy': list(metrics_vit.values_list('accuracy', flat=True)),
        'loss': list(metrics_vit.values_list('loss', flat=True))
    }

    cnn_data = {
        'epochs': list(metrics_cnn.values_list('epoch', flat=True)),
        'accuracy': list(metrics_cnn.values_list('accuracy', flat=True)),
        'loss': list(metrics_cnn.values_list('loss', flat=True))
    }
        # Obtener todos los registros ordenados por class_label
    vit_matrices = Confusion_Matrix.objects.filter(brand='Vit').order_by('class_label')
    cnn_matrices = Confusion_Matrix.objects.filter(brand='CNN').order_by('class_label')

    # Filtrar registros para obtener solo el primer de cada class_label
    vit_unique_matrices = []
    seen_classes = set()  # Set para llevar registro de las clases ya vistas
    for matrix in vit_matrices:
        if matrix.class_label not in seen_classes:
            vit_unique_matrices.append(matrix)
            seen_classes.add(matrix.class_label)

    cnn_unique_matrices = []
    seen_classes = set()  # Reiniciar el set para las matrices de CNN
    for matrix in cnn_matrices:
        if matrix.class_label not in seen_classes:
            cnn_unique_matrices.append(matrix)
            seen_classes.add(matrix.class_label)

    # Limitar a los primeros 4 elementos si es necesario
    vit_matrices = vit_unique_matrices[:4]
    cnn_matrices = cnn_unique_matrices[:4]
    
    # Pasar los datos al template
    context = {
        'vit_data_json': json.dumps(vit_data),
        'cnn_data_json': json.dumps(cnn_data),
        'title': 'Comparación ViT vs CNN',
        'vit_matrices': vit_matrices,
        'cnn_matrices': cnn_matrices
    }
    
    print(f"✅ Datos para ViT y CNN cargados correctamente {context}")
    return render(request, 'SigatokaDetectionSystem/SigatokaComparison.html', context)