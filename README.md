# Pizza vs Steak Classifier 🍕

This project uses TensorFlow and Keras to build two image classification models that distinguish between images of **pizza** and **steak**.

## Models
- **Model 1:** Fully connected neural network (Dense)
- **Model 2:** Convolutional Neural Network (CNN)

## Dataset
- Source: [`pizza_steak.zip`](https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip)
- Two classes: `pizza` and `steak`
- Data is preprocessed using `ImageDataGenerator` with normalization.

## Results
- The CNN model significantly outperforms the Dense model.
- Model evaluation and image predictions are visualized using Matplotlib.

## Requirements
- Python
- TensorFlow
- Matplotlib
