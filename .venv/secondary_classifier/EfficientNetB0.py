import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# EfficientNetB0 modelini yükle
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True  # Fine-tuning açık

# Modeli oluştur
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 sınıf: dog ve not_dog
])

# Modeli derle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentasyon (veri artırma)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Dizinler
train_dir = 'C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/dataset/images/train'
validation_dir = 'C:/Users/Sibel Kaya/Desktop/DogIdwithYOLO/.venv/dataset/images/valid'

# Verileri yükle
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Modeli eğit
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Modeli kaydet
model.save('dog_human_classifier_model.h5')

print("Model başarıyla eğitildi ve kaydedildi! 🎯")
