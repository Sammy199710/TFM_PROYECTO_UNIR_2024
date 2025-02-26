import os

import django

# Configurar Django antes de importar modelos
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ComparativaSigatoka.settings")
django.setup()

print(os.getcwd())

import tensorflow as tf
print(f"🔍 Eager Execution está activado: {tf.executing_eagerly()}")
tf.config.run_functions_eagerly(True)
print("✅ Eager Execution activado correctamente.")
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')  # Usa float16 en GPU para ahorrar memoria
mixed_precision.set_global_policy(policy)

from SigatokaDetectionSystem.TensorflowIa.models.cnn_model import create_cnn_model
from SigatokaDetectionSystem.TensorflowIa.models.vit_model import create_vit_model
from SigatokaDetectionSystem.TensorflowIa.utils.preprocessing import create_dataset
from SigatokaDetectionSystem.TensorflowIa.utils.RegistroValores import *
from sklearn.metrics import confusion_matrix, classification_report


# 🔹 Configurar TensorFlow para usar la GPU NVIDIA y limitar uso de memoria 🔹
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]  # Limita a 6GB de VRAM
        )
        print("✅ Memoria GPU limitada a 6GB para evitar cuelgues.")
    except RuntimeError as e:
        print(f"⚠️ Error al configurar límite de memoria: {e}")

    """try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Habilita uso dinámico de memoria
        print("✅ TensorFlow configurado para uso eficiente de memoria en GPU.")
    except RuntimeError as e:
        print(f"⚠️ Error al configurar memoria dinámica: {e}")"""


tf.config.experimental.enable_tensor_float_32_execution(True)

print("🔍 TensorFlow reconoce GPU:", tf.config.list_physical_devices('GPU'))
print("🔍 TensorFlow está compilado con CUDA:", tf.test.is_built_with_cuda())
print("🔍 TensorFlow está compilado con cuDNN:", tf.test.is_built_with_gpu_support())
print("🔍 XLA activado:", tf.config.optimizer.get_jit())

# Configuración
BATCH_SIZE = 4
IMG_SIZE = (224, 224)
FOLDS = 5  # Para validación cruzada

# Obtener ruta de datos
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Cargar datos
train_data, _ = create_dataset("train")

# Verificar dataset cargado
print(f"🔎 train_data es de tipo: {type(train_data)}")

# Intentar obtener el primer batch
try:
    first_batch = next(iter(train_data))
    print(f"✅ El primer batch de train_data se cargó correctamente.")
    print(f"📸 Tamaño del batch de imágenes: {first_batch[0].shape}")  # Si usa imágenes
    print(f"🎯 Tamaño del batch de etiquetas: {first_batch[1].shape}")  # Si usa etiquetas
except Exception as e:
    print(f"🚨 Error al cargar el primer batch: {e}")

val_data, _ = create_dataset("val")
test_data, _ = create_dataset("test")

train_labels = train_data.labels  # ⚡ Obtener etiquetas sin iterar
class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Callbacks para optimización
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Validación cruzada
kf = KFold(n_splits=FOLDS, shuffle=True)
cnn_accuracies, vit_accuracies = [], []
intEpochs=5


for fold, (train_idx, val_idx) in enumerate(kf.split(train_labels)):
    intAcumulador = fold + 1
    print(f"\n📌 Fold {intAcumulador}/{FOLDS}")
    
    # Entrenar modelo CNN
    cnn_model = create_cnn_model()
    cnn_history = cnn_model.fit(train_data, validation_data=val_data, epochs=intEpochs, class_weight=class_weight_dict, callbacks=[reduce_lr])
    
    cnn_loss, cnn_acc = cnn_model.evaluate(test_data)
    preds_cnn = cnn_model.predict(test_data)
    pred_labels_cnn = np.argmax(preds_cnn, axis=1)  # Para clasificación con softmax
    true_labels = test_data.labels # Ya tienes las etiquetas verdaderas
    # Registrar matriz de confusión para CNN
    registrar_matriz_confusion(true_labels, pred_labels_cnn, 'CNN', average_type="binary")

    # Matriz de confusión
    print(f"=== CNN - Fold {intAcumulador} - Matriz de confusión ===")
    print(confusion_matrix(true_labels, pred_labels_cnn))

    # Reporte de clasificación con precisión, recall, F1, soporte, etc.
    print(f"=== CNN - Fold {intAcumulador} - Classification Report ===")
    print(classification_report(true_labels, pred_labels_cnn))
    cnn_accuracies.append(cnn_acc)
    registrar_metrica(cnn_history, "CNN", intEpochs,intAcumulador)

    # Entrenar modelo ViT
    vit_model = create_vit_model()
    #Convertir la imagenes a float32 
    train_data = train_data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    val_data = val_data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    test_data = test_data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    vit_history = vit_model.fit(train_data, validation_data=val_data, epochs=5, class_weight=class_weight_dict, callbacks=[reduce_lr])
    vit_loss, vit_acc = vit_model.evaluate(test_data)
    preds_vit = vit_model.predict(test_data)
    pred_labels_vit = np.argmax(preds_vit, axis=1)
    true_labels = test_data.labels
    registrar_matriz_confusion(true_labels, pred_labels_vit, 'Vit',  average_type="binary")

    print(f"=== ViT - Fold {intAcumulador} - Matriz de confusión ===")
    print(confusion_matrix(true_labels, pred_labels_vit))

    print(f"=== ViT - Fold {intAcumulador} - Classification Report ===")
    print(classification_report(true_labels, pred_labels_vit))
    vit_accuracies.append(vit_acc)
    registrar_metrica(vit_history, "ViT", intEpochs,intAcumulador)

print("\n✅ Validación cruzada completada.")

fltPrecisionCnn = np.mean(cnn_accuracies)
fltPrecisionVit = np.mean(vit_accuracies)
print(f"📊 Precisión promedio CNN: {fltPrecisionCnn:.4f}")
registrar_analitica(None, 'CNN', fltPrecisionCnn)
print(f"📊 Precisión promedio ViT: {fltPrecisionVit:.4f}")
registrar_analitica(None, 'ViT', fltPrecisionVit)

# Evaluación final con dataset de prueba
print("\n🔍 Evaluando modelos en el conjunto de test...")

cnn_history_final = cnn_model.fit(train_data, validation_data=val_data, epochs=10, class_weight=class_weight_dict, callbacks=[reduce_lr])
cnn_model.evaluate(test_data)
preds_cnn_final = cnn_model.predict(test_data)
pred_labels_cnn_final = np.argmax(preds_cnn_final, axis=1)
true_labels = test_data.labels
# Registra la matriz de confusión
registrar_matriz_confusion(true_labels, pred_labels_cnn_final, 'CNN', average_type="binary")
print("=== CNN - FINAL - Matriz de confusión ===")
print(confusion_matrix(true_labels, pred_labels_cnn_final))
print("=== CNN - FINAL - Classification Report ===")
print(classification_report(true_labels, pred_labels_cnn_final))
registrar_metrica(cnn_history_final, "CNN", 10, None)

vit_history_final = vit_model.fit(train_data, validation_data=val_data, epochs=10, class_weight=class_weight_dict, callbacks=[reduce_lr])
vit_model.evaluate(test_data)
preds_vit_final = vit_model.predict(test_data)
pred_labels_vit_final = np.argmax(preds_vit_final, axis=1)
registrar_matriz_confusion(true_labels, pred_labels_vit_final, 'Vit', average_type="binary")
print("=== ViT - FINAL - Matriz de confusión ===")
print(confusion_matrix(true_labels, pred_labels_vit_final))
print("=== ViT - FINAL - Classification Report ===")
print(classification_report(true_labels, pred_labels_vit_final))
registrar_metrica(vit_history_final, "ViT", 10, None)