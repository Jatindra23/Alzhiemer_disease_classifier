import logging
import numpy as np
from keras.models import load_model

# from keras.preprocessing import image
from keras.utils import load_img
from keras.utils import img_to_array
import os


class PredictionPipeline:
    # Class variable to hold the model to avoid reloading it multiple times
    model = None

    def __init__(self, filename):
        self.filename = filename
        if PredictionPipeline.model is None:
            PredictionPipeline.model = self.load_model()

    def load_model(self):
        """
        Loads the pre-trained machine learning model.
        """
        try:
            # model_path = os.path.join("artifacts", "training", "model.keras")
            model_path = "artifacts\training\model.keras"
            model = load_model(model_path)
            logging.info(f"Model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict(self):
        try:
            # Load and preprocess image
            test_image = load_img(self.filename, target_size=(224, 224))
            test_image = img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Normalize if required
            logging.debug("Image loaded and preprocessed successfully.")

            # Make prediction
            prediction_prob = PredictionPipeline.model.predict(test_image)
            result = np.argmax(prediction_prob, axis=1)
            logging.info(f"Raw prediction output: {prediction_prob}")
            logging.info(f"Argmax result: {result}")

            # Decode prediction
            if result[0] == 1:
                prediction = "Tumor"
            else:
                prediction = "Normal"

            logging.info(f"Decoded prediction: {prediction}")
            return {"prediction": prediction}
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
