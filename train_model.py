import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import tf2onnx
import numpy as np
import matplotlib.pyplot as plt

# CONFIG
DATASET_DIR = "Final_Submission_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Keep it small for speed

def train_and_export():
    # 1. Load Data
    print("Loading Data...")
    train_dir = os.path.join(DATASET_DIR, 'Train')
    val_dir = os.path.join(DATASET_DIR, 'Validation')
    test_dir = os.path.join(DATASET_DIR, 'Test')

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    val_generator = test_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    # 2. Build Model (Transfer Learning)
    print("Building MobileNetV2 Model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(train_generator.class_indices), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=preds)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train
    print("Training...")
    history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

    # 4. Evaluate & Generate Metrics
    print("Evaluating...")
    results = model.evaluate(test_generator)
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    
    # Generate Confusion Matrix & Report
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print("Classification Report:")
    report = classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys())
    print(report)
    
    # Save Report to text file
    with open("model_results.txt", "w") as f:
        f.write(f"Test Accuracy: {results[1]*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nModel Size: MobileNetV2 (approx 14MB)\n")
        f.write("Platform: TensorFlow / Keras\n")

    # 5. Export to ONNX
    print("Exporting to ONNX...")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    output_path = "model.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"ONNX model saved to {output_path}")

if __name__ == "__main__":
    train_and_export()