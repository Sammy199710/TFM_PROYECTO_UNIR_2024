from SigatokaDetectionSystem.models import Metric, Analityc, Confusion_Matrix  # Asegúrate de importar el modelo Metric de tu aplicación Django
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from SigatokaDetectionSystem.models import Confusion_Matrix  # Ajusta la ruta de import según tu estructura


def registrar_metrica(history, strBandera, epoch_brand, intAcumulador = None):
    """
    Registra las métricas de todas las épocas en la base de datos.
    
    history: Objeto de historial de entrenamiento del modelo
    strBandera: Nombre del modelo (CNN o ViT)
    """
    for epoch, (accuracy, loss) in enumerate(zip(history.history['accuracy'], history.history['loss']), start=1):
        print(f"\n✅ registrar_metrica epoch={epoch}, accuracy={accuracy}, loss={loss}, brand={strBandera}, epoch_brand={epoch_brand}")
        metric = Metric(
            epoch=epoch,
            accuracy=accuracy, 
            loss=loss,
            brand=strBandera, 
            epoch_brand=epoch_brand, 
            folds= intAcumulador if intAcumulador is not None else None )
        metric.save()  # Guarda cada época en la base de datos


def registrar_analitica( descripcion, strBandera, promedio= None):
    """
    Registra las métricas de todas las épocas en la base de datos.
    
    history: Objeto de historial de entrenamiento del modelo
    strBandera: Nombre del modelo (CNN o ViT)
    """
    print(f"\n✅ registrar_analitica descripcion={descripcion}, promedio={promedio}")
    objAnalitic = Analityc(brand=strBandera, description= descripcion,average =  promedio if promedio is not None else None )
    objAnalitic.save()  # Guarda cada época en la base de datos


def registrar_matriz_confusion(true_labels, pred_labels, strBandera, average_type="binary"):
    """
    Guarda en la base de datos las métricas de cada clase (0, 1, etc.)
    y el promedio global definido por 'average_type'.

    Parámetros:
    - true_labels: array/list de etiquetas verdaderas
    - pred_labels: array/list de etiquetas predichas
    - average_type: 'binary', 'macro', 'micro', 'weighted', etc.
    """

    # 1. Guardar métricas por cada clase (por ejemplo, 0 y 1)
    classes = np.unique(true_labels)  # si es binario, típicamente [0, 1]
    for c in classes:
        # Calculamos p, r, f1 para la clase c
        p_c = precision_score(true_labels, pred_labels, labels=[c], average='binary')
        r_c = recall_score(true_labels, pred_labels, labels=[c], average='binary')
        f1_c = f1_score(true_labels, pred_labels, labels=[c], average='binary')
        support_c = int(np.sum(true_labels == c))

        # Guardar en la base
        cm_class = Confusion_Matrix(
            precision=p_c,
            recall=r_c,
            f1_score=f1_c,
            support=support_c,
            class_label=str(c),
            brand=strBandera
        )
        cm_class.save()
        print(f"Clase={c}: p={p_c:.2f}, r={r_c:.2f}, f1={f1_c:.2f}, soporte={support_c}, bandera={strBandera}")

    # 2. Guardar el promedio (según average_type)
    p_avg = precision_score(true_labels, pred_labels, average=average_type)
    r_avg = recall_score(true_labels, pred_labels, average=average_type)
    f1_avg = f1_score(true_labels, pred_labels, average=average_type)
    support_avg = len(true_labels)  # total de muestras

    cm_avg = Confusion_Matrix(
        precision=p_avg,
        recall=r_avg,
        f1_score=f1_avg,
        support=support_avg,
        class_label=f"AVG-{average_type}",
        brand=strBandera
    )
    cm_avg.save()
    print(f"Promedio {average_type}: p={p_avg:.2f}, r={r_avg:.2f}, f1={f1_avg:.2f}, soporte={support_avg}, bandera={strBandera}")