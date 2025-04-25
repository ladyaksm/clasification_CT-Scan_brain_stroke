# 🧠 Stroke Classification from CT Scan Images
This project focuses on building a Convolutional Neural Network (CNN) model to classify CT scan brain images into stroke types: **Ischemic**, **Bleeding**, or **Normal**. The goal is to assist early detection and decision-making during the golden period of stroke treatment.

## 📂 Project Structure
classification_CT-Scan_brain_stroke/
│
├── models/                    # SavedModel, TF-Lite, TFJS exports
├── notebook/                  # Jupyter Notebook for training & evaluation
├── README.md
└── requirements.txt

## 🔍 Problem Statement
Stroke is a leading cause of death and disability. Early identification of the stroke type is critical for timely intervention. This project uses **deep learning** to detect:

- 🧠 **Ischemic Stroke**
- 🩸 **Bleeding Stroke**
- ✅ **Normal (No Stroke)**



## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib**
- **CNN Model (Conv2D, MaxPooling, BatchNorm, Dropout)**
- **Deployment formats:** SavedModel, TFLite, TensorFlow.js (TFJS)



## 🧠 Model Architecture
python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])




## 📊 Model Performance
- ✅ Accuracy: >85% (Target)
- 🧪 Evaluated on test set with balanced stroke classes
- 📉 Trained with data augmentation and early stopping

## 🚀 Model Deployment

### ✅ SavedModel
```python
model.save('models/stroke_model.keras')
```

### 📱 TFLite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/stroke_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 🌐 TensorFlow.js (TFJS)
```bash
tensorflowjs_converter --input_format keras stroke_model.keras models/tfjs_model/
```

## 📁 Dataset Info
Dataset is split into:
- `train/`  
- `val/`  
- `test/`  

With balanced classes of:
- Bleeding
- Ischemia
- Normal (optional)


## 📌 How to Run
1. Clone this repo  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model or load saved model
4. Export to SavedModel / TFLite / TFJS
5. Deploy to desired platform (mobile/web/server)



## 👤 Author

**Ladya KSM**  
_A student exploring medical imaging with machine learning for real-world impact._

