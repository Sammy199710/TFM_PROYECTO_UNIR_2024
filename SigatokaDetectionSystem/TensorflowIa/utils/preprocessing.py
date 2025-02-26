import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_dataset(data_dir, batch_size=4, img_size=(224, 224)):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    dataset_path = os.path.join(base_path, data_dir)

    print(f"📌 Cargando dataset desde: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ ERROR: La carpeta {dataset_path} no existe.")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_generator, val_generator