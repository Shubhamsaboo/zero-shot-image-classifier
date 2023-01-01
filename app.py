# Description: This is a Streamlit app that uses the CLIP model to perform zero-shot image classification.

# Import the necessary libraries
from PIL import Image
import requests
import streamlit as st
from transformers import CLIPProcessor, CLIPModel

# Set up the Streamlit app
st.title("🔎 Zero-Shot Image Classifier")

# Get the image from the user
upload_type = st.select_slider('Upload an Image or provide URL', ['Image','URL'])

if upload_type == 'Image':
    inp_image = st.file_uploader("Upload an image:", type=['png','jpg','jpeg'])
    image = Image.open(inp_image)
    st.image(image, width=500)
elif upload_type == 'URL':
    inp_image = st.text_input("Enter the URL of the image:", "http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(requests.get(inp_image, stream=True).raw)
    st.image(image, width=500)


# Get the class names from the user
class_names = st.text_input("Enter the names of the classes, separated by commas:")
classes = [class_name.strip() for class_name in class_names.split(",")]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Run the model and display the results 
if st.button("Run Model"):
    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    for i, class_name in enumerate(classes):
        st.write(f"Probability that the image is a photo of {class_name}: {probs[0][i]:.2f}")