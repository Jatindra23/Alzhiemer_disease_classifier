from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Alzimer_disease_classifier.utils.common import decodeImage
from Alzimer_disease_classifier.pipeline.prediction import PredictionPipeline
from Alzimer_disease_classifier.logger import logging
from PIL import Image
from werkzeug.utils import secure_filename


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if allowed, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_image(file_stream):
    """
    Validates whether the uploaded file is a valid image.

    Args:
        file_stream (BytesIO): The file stream of the uploaded image.

    Returns:
        bool: True if valid image, False otherwise.
    """
    try:
        image = Image.open(file_stream)
        image.verify()  # Verifies that it is, in fact, an image
        return True
    except Exception:
        return False 


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")  # for DVC
    return "Training done successfully!"


# @app.route("/predict", methods=["POST"])
# @cross_origin()
# def predictRoute():
#     image = request.json["image"]
#     decodeImage(image, clApp.filename)
#     result = clApp.classifier.predict()
#     return jsonify(result)
@app.route("/predict", methods=["POST"])
def predictRoute():
    try:
        logging.info("Received a prediction request.")

        if "image" not in request.files:
            logging.warning("No image part in the request.")
            return jsonify({"error": "No image part in the request"}), 400

        image = request.files["image"]

        if image.filename == "":
            logging.warning("No image selected for uploading.")
            return jsonify({"error": "No image selected for uploading"}), 400

        if not allowed_file(image.filename):
            logging.warning(f"Unsupported file type: {image.filename}")
            return jsonify({"error": "Unsupported file type."}), 400

        # Validate image content
        image_stream = image.stream
        if not is_image(image_stream):
            logging.warning("Uploaded file is not a valid image.")
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        # Reset stream position after verification
        image_stream.seek(0)

        # Secure the filename
        secure_name = secure_filename(image.filename)
        logging.debug(f"Secured filename: {secure_name}")

        # Save the uploaded image
        decodeImage(image, clApp.filename)
        logging.info(f"Image saved successfully at {clApp.filename}.")

        # Perform prediction
        logging.info("Starting prediction...")
        result = clApp.classifier.predict()
        logging.info(f"Prediction result: {result}")

        return jsonify(result)
    except Exception as e:
        logging.info(f"Prediction Error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500


if __name__ == "__main__":
    clApp = ClientApp()

    app.run(host="0.0.0.0", port=8080)  # for AWS
    # app.run(host="0.0.0.0", port=8000)
    
