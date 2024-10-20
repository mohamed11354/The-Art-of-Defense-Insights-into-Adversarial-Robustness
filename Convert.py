import tensorflow as tf
efficientnet=tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True,
        classes=1000,
        classifier_activation='softmax',
        input_shape=(224,224,3),
        weights=None)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input, input_shape=(224, 224, 3)))
model.add(efficientnet)


model.load_weights('D:\projects\\advarsrial_research\\Final\\efficientnetb0\\checkpoint\\adv_efficientnet')

model.compile(optimizer="sgd",loss = "sparse_categorical_crossentropy",metrics=['sparse_categorical_accuracy'])

model.save_weights("D:\projects\\advarsrial_research\\Final\\efficientnetb0\\checkpoint.h5")
