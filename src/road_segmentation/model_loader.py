#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-
import tensorflow as tensorflow

def load_model(model_path):
    # Load the pre-trained model
    model = tensorflow.keras.models.load_model(model_path)
    return model

def predict(model, image):
    # Predict segmentation mask from image
    prediction = model.predict(image)
    return prediction

