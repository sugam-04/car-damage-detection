# Import necessary libraries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model directly using load_model
model_loaded = load_model("Car_detection.model")

# Function to make predictions
def predict_car_damage(image_path):
    # Load the image
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    preprocessed_image = preprocess_input(image_array.reshape(1, 224, 224, 3))

    # Make predictions
    predictions = model_loaded.predict(preprocessed_image)

    # Convert predictions to class labels
    class_labels = ["00-damage", "01-whole"]
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

# Streamlit app code
def main():
    st.title("Car Damage Detection")
    st.write("Upload an image of a car to check if it's damaged or not.")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make predictions and display the result
        predicted_class_label = predict_car_damage(uploaded_image)
        if predicted_class_label == "00-damage":
            st.write("The car is damaged.")
        else:
            st.write("The car is not damaged.")

# Run the app
if __name__ == "__main__":
    main()
