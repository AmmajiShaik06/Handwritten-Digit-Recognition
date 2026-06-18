# Handwritten Digit Recognition Using CNN

## Project Overview

Handwritten Digit Recognition is a deep learning project that classifies handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset, which contains grayscale images of handwritten digits.

## Objective

The objective of this project is to develop a machine learning model capable of accurately recognizing handwritten digits and classifying them into one of the ten digit classes (0–9).

## Dataset

**MNIST Dataset**

* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Number of Classes: 10 (Digits 0–9)

## Technologies Used

* Python
* Jupyter Notebook
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

## Project Workflow

1. Load MNIST Dataset
2. Data Visualization
3. Data Preprocessing
4. Build CNN Model
5. Train the Model
6. Evaluate Performance
7. Generate Predictions
8. Save Trained Model
9. Visualize Results

## CNN Architecture

* Conv2D Layer (32 Filters)
* MaxPooling2D Layer
* Flatten Layer
* Dense Layer (128 Neurons)
* Output Layer (10 Neurons, Softmax)

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Results

The CNN model achieved high classification accuracy on the MNIST test dataset and successfully recognized handwritten digits.

## Project Structure

Handwritten_Digit_Recognition/

├── digit_recognition.ipynb

├── README.md

├── dataset/

│   ├── X_train.npy

│   ├── y_train.npy

│   ├── X_test.npy

│   └── y_test.npy

├── model/

│   └── mnist_cnn.h5

├── screenshots/

│   ├── accuracy.png

│   ├── confusion_matrix.png

│   └── predictions.png

└── report/

```
├── classification_report.txt

└── report.pdf
```

## Conclusion

This project demonstrates the effectiveness of Convolutional Neural Networks in image classification tasks. The trained model accurately identifies handwritten digits and can be extended for real-world optical character recognition application.
