import streamlit as st
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- LOAD MODEL ----------------
model = load_model("fruit_model.h5")

# labels.json is a LIST (not dictionary)
with open("labels.json") as f:
    classes = json.load(f)

st.title("🍎 Fruit Ripeness Detection System")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Upload Image", "Live Camera"]
)

# =====================================================
# 📷 MODE 1 : IMAGE UPLOAD
# =====================================================
if mode == "Upload Image":

    st.header("Upload Fruit Image")

    file = st.file_uploader("Choose Image")

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess image (same as training)
        img = np.array(img)
        img = cv2.resize(img,(224,224))
        img = img/255.0
        img = np.reshape(img,(1,224,224,3))

        # Predict
        pred = model.predict(img)
        label = classes[np.argmax(pred)]

        st.success(f"Prediction: {label}")

# =====================================================
# 🎥 MODE 2 : REAL-TIME CAMERA
# =====================================================
if mode == "Live Camera":

    st.header("Real-Time Camera Detection")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Camera not working")
            break

        img = cv2.resize(frame,(224,224))
        img = img/255.0
        img = np.reshape(img,(1,224,224,3))

        pred = model.predict(img)
        label = classes[np.argmax(pred)]

        # Show prediction on frame
        cv2.putText(frame,label,(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()
