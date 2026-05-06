code = '''
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

IMG_SIZE = 80
CATEGORIES = ["real", "fake"]

st.title("AI Image Detector")

st.write(
    "Upload an image and the model will classify it as real or fake."
)

# ---------------------------------
# DOWNLOAD MODEL FROM HUGGING FACE
# ---------------------------------

MODEL_PATH = hf_hub_download(
    repo_id="LiamReardon01/ai-image-detector-model",
    filename="ai_image_detector.keras"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

st.success("Model loaded successfully.")

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# PREDICTION
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    IMG = image.resize((IMG_SIZE, IMG_SIZE))

    X_RAW = np.array(IMG).reshape(
        -1,
        IMG_SIZE,
        IMG_SIZE,
        3
    )

    X_RAW = X_RAW / 255.0

    if st.button("Run Prediction"):

        Y = model.predict(X_RAW)

        st.write("Raw model output:")
        st.write(Y)

        Y = list(Y[0])

        maxValue = max(Y)

        theIndex = Y.index(maxValue)

        prediction = CATEGORIES[theIndex]

        st.subheader("Result")

        if prediction == "fake":

            st.error(
                "This is likely AI-generated / fake."
            )

        else:

            st.success(
                "This is likely real."
            )

        st.write(
            "Confidence:",
            round(float(maxValue) * 100, 2),
            "%"
        )
'''

with open("app.py", "w") as f:
    f.write(code)

