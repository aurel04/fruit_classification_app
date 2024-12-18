import streamlit as st
import pandas as pd
from util import load_fruit_model, preprocess_image, predict_fruit_class

# Paths to model and class indices
# Model to be modified after training finished
model_path = "model/ResNet50V2.h5"
class_indices_path = "model/class_indices.json"

model, class_labels = load_fruit_model(model_path, class_indices_path)

st.title("Fruit Image Classification Using Hybrid CNN")
st.caption("Upload a fruit image file!")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)

    preprocessed_image, preproim = preprocess_image(uploaded_file)

    st.image(preproim, use_container_width=True)
    
    predicted_class_label, confidence_scores = predict_fruit_class(model, class_labels, preprocessed_image)

    st.header(f"Prediction : {predicted_class_label}")

    confidence_scores_df = pd.DataFrame.from_dict(confidence_scores, orient="index", columns=["Confidence Score (%)"]).reset_index().rename(columns={"index": "Class"})
    confidence_scores_df = confidence_scores_df.sort_values(by="Confidence Score (%)", ascending=False).head(5)
    
    show_confidence_button = st.button("Show Top 5 Confidence Scores")
    
    if show_confidence_button:
        st.subheader("Top 5 Confidence Scores")
        st.table(confidence_scores_df)