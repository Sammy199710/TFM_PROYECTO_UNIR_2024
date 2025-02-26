import tensorflow as tf
import tensorflow_hub as hub

def create_vit_model():
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    #vit_layer = hub.KerasLayer(vit_url, trainable=True, dynamic=True)  # Activamos el fine-tuning
    vit_layer = hub.KerasLayer(vit_url, trainable=True)

    inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)

    # 🔹 Asegurar que las imágenes sean float32 antes de ViT
    x = tf.keras.layers.Lambda(lambda x: tf.image.convert_image_dtype(x, tf.float32))(inputs)

    #x = tf.keras.applications.imagenet_utils.preprocess_input(inputs, mode="tf")
    # Preprocesamiento dentro de una Lambda
    x = tf.keras.layers.Lambda(
        lambda x: tf.keras.applications.imagenet_utils.preprocess_input(x, mode="tf")
    )(x)
    # 🔹 Aplicar ViT
    x = vit_layer(x)
    # Envolver la llamada al layer de TF Hub en una Lambda, forzando el argumento training
    #x = tf.keras.layers.Lambda(lambda x: vit_layer(x, training=True))(x)
    #vit_layer = tf.keras.layers.Lambda(lambda x: hub.KerasLayer(vit_url, trainable=True)(x))
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"],
                  jit_compile=False)
    return model
