# 🍓 Fruit Classification System
This project is a fruit image classification web app built using Streamlit and Convolutional Neural Networks (CNNs). It supports two hybrid model configurations trained on different datasets (Ubaya-IFDS3000 and Ubaya-IFDS5000).

## 🔍 Overview
This system allows user to upload an image which then will be categorized into different fruit type by the system.

## 🧰 Techs in use
- **Model 1**: Hybrid CNN VGG16 + ResNet50V2 (trained on 3000 images)
- **Model 2**: Hybrid CNN ResNet50V2 + InceptionV3 (trained on 5000 images)
- **Framework**: Streamlit
- **Progamming Language**: python

## ⚙️ Requirements
- Python 3.8+
- TensorFlow
- Streamlit
- Pillow
- NumPy
- Pandas

Install dependencies with:
pip install -r requirements.txt

# 1. Clone the repository

# 2. Run the Streamlit app:
streamlit run app.py

# 3. Install Node dependencies
npm install

# 4. Open the local web link provided
