import streamlit as st
import os
from PIL import Image
from werkzeug.utils import secure_filename
from Alzimer_disease_classifier.utils.common import decodeImage
from Alzimer_disease_classifier.pipeline.prediction import PredictionPipeline
from Alzimer_disease_classifier.logger import logging

# Set environment variables
os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image(file_stream):
    """
    Validates whether the uploaded file is a valid image.
    """
    try:
        img = Image.open(file_stream)
        img.verify()  # Verifies that it is an image
        return True
    except Exception:
        return False 

class ClientApp:
    def __init__(self):
        # Local filename to store the uploaded image
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

def main():
    st.title("Alzheimer Disease Classifier")
    
    # Initialize our client application
    clApp = ClientApp()
    
    # Sidebar navigation for selecting an action
    action = st.sidebar.selectbox("Select Action", ["Predict", "Train"])
    
    if action == "Train":
        st.header("Train the Model")
        st.write("Click the button below to start training.")
        
        if st.button("Start Training"):
            # Ensure main.py is your training script, not this file.
            os.system("python main.py")
            st.success("Training done successfully!")
    
    elif action == "Predict":
        st.header("Upload an Image for Prediction")
        
        # File uploader for the image
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])
        
        if uploaded_file is not None:
            if not allowed_file(uploaded_file.name):
                st.error("Unsupported file type.")
            else:
                # Validate the file to ensure it is an image
                if not is_image(uploaded_file):
                    st.error("Uploaded file is not a valid image.")
                else:
                    # Reset the file pointer after validation
                    uploaded_file.seek(0)
                    
                    # Display the uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Secure the filename (optional in Streamlit)
                    secure_name = secure_filename(uploaded_file.name)
                    logging.debug(f"Secured filename: {secure_name}")
                    
                    # Save the image to disk using the defined filename in ClientApp
                    # Optionally, you can call decodeImage if needed:
                    # decodeImage(uploaded_file, clApp.filename)
                    with open(clApp.filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("Image saved successfully!")
                    
                    # Trigger prediction on button click
                    if st.button("Predict"):
                        try:
                            logging.info("Starting prediction...")
                            result = clApp.classifier.predict()
                            logging.info(f"Prediction result: {result}")
                            
                            st.subheader("Prediction Result")
                            st.write(result)
                        except Exception as e:
                            logging.error(f"Prediction Error: {e}")
                            st.error("An error occurred during prediction.")

if __name__ == "__main__":
    main()
