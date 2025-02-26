import os
import shutil
import random

# Obtener la ruta absoluta del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
print(f"📌 BASE_DIR detectado: {BASE_DIR}")

# Definir rutas absolutas del dataset
data_dir = os.path.join(BASE_DIR, "data")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Verificar que las carpetas existen
print(f"📌 Verificando existencia de {train_dir}/healthy")
print(f"📌 Verificando existencia de {train_dir}/infected")

if not os.path.exists(train_dir):
    print(f"❌ ERROR: La carpeta de entrenamiento {train_dir} no existe.")
    exit()

for category in ["healthy", "infected"]:
    category_path = os.path.join(train_dir, category)
    if not os.path.exists(category_path):
        print(f"❌ ERROR: La carpeta {category_path} no existe.")
        exit()

# Crear carpetas si no existen
for category in ["healthy", "infected"]:
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Definir proporciones del dataset
val_split = 0.10  # 10% para validación
test_split = 0.10  # 10% para prueba

# Reubicar imágenes en sus respectivos conjuntos
for category in ["healthy", "infected"]:
    category_path = os.path.join(train_dir, category)
    img_paths = os.listdir(category_path)

    if len(img_paths) == 0:
        print(f"⚠️ ADVERTENCIA: No hay imágenes en '{category_path}'. Saltando esta categoría...")
        continue

    random.shuffle(img_paths)  # Mezclar aleatoriamente

    num_total = len(img_paths)
    num_val = int(num_total * val_split)
    num_test = int(num_total * test_split)

    val_imgs = img_paths[:num_val]
    test_imgs = img_paths[num_val:num_val + num_test]

    # Mover imágenes a validación
    for img in val_imgs:
        shutil.move(os.path.join(category_path, img), os.path.join(val_dir, category, img))

    # Mover imágenes a prueba
    for img in test_imgs:
        shutil.move(os.path.join(category_path, img), os.path.join(test_dir, category, img))

    print(f"✅ {num_val} imágenes movidas a '{val_dir}/{category}'")
    print(f"✅ {num_test} imágenes movidas a '{test_dir}/{category}'")

print("✅ División de datos completada correctamente.")