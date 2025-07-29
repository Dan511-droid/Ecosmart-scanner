import tensorflow as tf
from tensorflow import keras #These libraries provide the deep learning framework and high-level API to build, train, and fine-tune the model.
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Used to perform data augmentation—this helps improve generalization by randomly transforming images during training.
from tensorflow.keras.applications import MobileNetV2 #A lightweight, pre-trained CNN (trained on ImageNet) that serves as the base model.
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #Layers that are added on top of the base model to adapt it to your specific classification task.
from tensorflow.keras.models import Model #A callback to automatically reduce the learning rate when the validation loss stops improving.
from tensorflow.keras.callbacks import ReduceLROnPlateau #Used for file path operations (ensuring folders exist, etc.).
import os

# 🔹 Define dataset path
dataset_path = os.path.join("dataset")  # Ensure this folder exists with subfolders for each class

# 🔹 Ensure dataset exists before training
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder '{dataset_path}' not found! Make sure you have 'Metal', 'Plastic', and 'Organic' folders inside.")

img_size = (150, 150)
batch_size = 32
num_classes = 4  # Adjust based on your categories

# 🔹 Data Augmentation for Better Generalization
# Normalizes the images (scales pixel values to 0–1) and applies random rotations, shifts, flips, and zooms to enhance the training data and reduce overfitting.
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 20% of data for validation
)

# 🔹 Load Training Data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 🔹 Print Class Indices to Ensure Label Order
print("Class indices assigned by TensorFlow:", train_data.class_indices)

# 🔹 Load Validation Data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#Pre-trained Base Model:
#Loads MobileNetV2 without its top (final) layers, using pre-trained ImageNet weights.
# 🔹 Use a Pretrained Model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

#Freeze Layers:
#Sets all layers in the base model as non-trainable so that only the newly added layers will learn initially.
# 🔹 Freeze base model layers (fine-tuning option below)
for layer in base_model.layers:
    layer.trainable = False

#Custom Classification Layers:
#GlobalAveragePooling2D: Reduces the spatial dimensions of the feature maps, summarizing them.
#Dense(128, 'relu'): Adds a fully connected layer with 128 neurons.
#Dense(num_classes, 'softmax'): Adds the final layer that outputs probabilities for each waste category.

# 🔹 Add Custom Classification Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(num_classes, activation='softmax')(x)
#Final Model:Combines the pre-trained base and the new custom layers into a single model.
# 🔹 Create the Final Model
model = Model(inputs=base_model.input, outputs=output_layer)

#######
#Compile:The model is compiled with the Adam optimizer, using categorical crossentropy loss (since labels are one-hot encoded) and tracking accuracy.

#Learning Rate Scheduler:ReduceLROnPlateau monitors validation loss and reduces the learning rate when improvements stagnate.

#Training:The model is trained on the training data and validated on the validation set for a set number of epochs.
#######
# 🔹 Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔹 Use ReduceLROnPlateau to Adjust Learning Rate Dynamically
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# 🔹 Train the Model
epochs = 10  # Increase if needed for better accuracy
model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[lr_scheduler])

#######
#Unfreeze Last 20 Layers:Allows some layers of the base model to be fine-tuned, adapting the pre-trained features to your specific dataset.

#Recompile with a Lower Learning Rate:Fine-tuning requires a smaller learning rate to adjust the weights gently.

#Additional Training:The model is trained for additional epochs to further improve accuracy.
#######
# 🔹 Optionally Fine-Tune Last 20 Layers for Better Accuracy
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 🔹 Fine-Tune Model with Additional Training
model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[lr_scheduler])

#Saving the Model:The trained model is saved to disk in the model folder as waste_model.h5.

#This file is then loaded by app.py when running the web application.
# 🔹 Save the Updated Model
os.makedirs('model', exist_ok=True)
model.save('model/waste_model.h5')
print("✅ Model trained and saved successfully as 'model/waste_model.h5'")
