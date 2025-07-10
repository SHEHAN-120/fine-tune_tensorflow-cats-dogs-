# 🐶🐱 Cat vs Dog Image Classifier with Transfer Learning

This project uses **TensorFlow 2.x** and **MobileNetV2** to build a binary image classifier that distinguishes between cats and dogs using **transfer learning** and **fine-tuning**.

---

## 📌 Project Summary

* ✅ **Model**: MobileNetV2 (pre-trained on ImageNet)
* 🎯 **Task**: Binary image classification (Cat vs Dog)
* 🧠 **Approach**: Transfer Learning + Fine-tuning
* 🖼️ **Input Size**: 128x128 RGB
* 💾 **Framework**: TensorFlow / Keras

---

## 🧠 Techniques Used

* ✅ **Transfer Learning** with `MobileNetV2` as base model
* ✅ Added a custom **Global Average Pooling layer** and **Dense output layer**
* ✅ **Binary cross-entropy** loss for 2-class classification
* ✅ **ImageDataGenerator** for efficient loading & rescaling
* ✅ Fine-tuning selected deeper layers of MobileNetV2
* ✅ Used **small learning rate** (0.0001) for stable training

---

## 🧪 Model Training

```python
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

* 📉 **Learning Rate**: 0.0001 to prevent large updates during fine-tuning
* 🔁 **Epochs**: Initially trained 8 epochs freezing base model, then 5 epochs fine-tuning
* 🐾 **Fine-tuning**: Unfroze layers after layer 100 to adapt model to dataset

---

## 📈 Evaluation and Observations

* Final validation accuracy reached **\~97.3%** after fine-tuning.
* However, **validation loss did not improve and slightly increased** after fine-tuning, indicating some **overfitting**.
* The model perfectly fits training data (accuracy \~1.0), but generalization is limited by dataset size and training setup.
* Suggested solutions:

  * Add **data augmentation** to increase data variability
  * Use **early stopping** to prevent overfitting
  * Fine-tune fewer layers or use smaller learning rates
  * Collect more labeled images


---

## ⚠️ Challenges Faced

* ⚖️ Choosing the right **base model** balancing accuracy and speed
* 🔧 Finding proper **fine-tuning depth** to improve performance without overfitting
* 📉 Handling **overfitting** due to limited dataset size and model capacity
* 🧪 Ensuring clean **train/validation split** for reliable evaluation
* 🖼️ Proper **image preprocessing** for consistent model input

---





