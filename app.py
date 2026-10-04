import gdown
import os

model_path = "model/hand_xray_model.pth"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)

    url = "https://drive.google.com/uc?id=1HCeuemTZsxrUOiHKT-NFVMkXCfNlNMFn"

    gdown.download(url, model_path, quiet=False)

import streamlit as st
import PIL
from PIL import Image, ImageOps
import numpy as np
import torchvision
import torch

import test

import warnings
warnings.filterwarnings("ignore")

conf_threshold = 0.5

#Main Page
st.title("Hand Fracture Detection using Deep Vision & AI")
st.write("This project predicts human hand fractures. To learn more about the project - Try it now!")

st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 10px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		border-radius: 2px 2px 2px 2px;
		gap: 10px;
        padding-left: 10px;
        padding-right: 10px;
		padding-top: 10px;
		padding-bottom: 10px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #7f91ad;
	}

</style>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Overview", "Test"])

with tab1:
    st.markdown("## Overview")

    st.write("Developed by: MANIKANTA REDDDY ")
    
    st.write("Model Used: ResNet")
    
    st.markdown("### System Architecture")
    network_img = "images/system_architecture.png"
    st.image(network_img)

    st.markdown("### Description")
    st.write("This project implements 'Hand Fracture Detection using Deep Vision & AI' using the Fast R-CNN (Region-based Convolutional Neural Network) architecture. The purpose of Faster R-CNN (Region-based Convolutional Neural Network) is to perform efficient and accurate object detection within images. It addresses the challenge of localizing and classifying objects of interest in images, a fundamental task in computer vision applications. Faster R-CNN achieves this by introducing a Region Proposal Network (RPN) to generate candidate object bounding boxes, which are then refined and classified by subsequent network components. By combining region proposal generation and object detection into a single unified framework, Faster R-CNN significantly improves detection accuracy while maintaining computational efficiency, making it suitable for real-time applications such as autonomous driving, surveillance, medical imaging, and more. The dataset containing hand X-ray images is prepared, with images and their corresponding labels loaded and augmented to resize and transform boundary boxes. The Faster R-CNN model is then instantiated, with a pre-trained ResNet-50 backbone and a custom classification layer for bone fracture detection. The training loop is executed over multiple epochs, optimizing the model's parameters using the Adam optimizer and minimizing the combined loss. The best-performing model is saved, and its performance is evaluated on the validation set, with the best model further tested on a separate test set. After training, the model's ability to detect fractures is evaluated by comparing its predictions with the actual fractures. This helps us understand how accurate the model is in finding fractures. Also, a confusion matrix is created to see how well the model performs for different types of fractures, providing insights into its overall performance. By combining the power of ResNet's feature extraction capabilities with Faster R-CNN's precise object localization and classification, the system can effectively detect bone fractures within medical images with improved accuracy and reliability.")

with tab2:
    st.markdown("## Upload Image & Test")
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def set_clicked():
        st.session_state.clicked = True

    st.button('Upload Image', on_click=set_clicked)
    if st.session_state.clicked:
        image = st.file_uploader("", type=["jpg", "png"])
        
        if image is not None:
            st.write("You selected the file:", image.name)
            
            resnet_model = test.get_model()
            device = torch.device('cpu')
            resnet_model.to(device)
            
            col1, col2 = st.columns(2)

            with col1:
                uploaded_image = PIL.Image.open(image)

                st.image(
                    uploaded_image,
                    caption="Uploaded Image",
                    width=300
                )

                
                to_tensor = torchvision.transforms.ToTensor()
                content = to_tensor(uploaded_image).unsqueeze(0)
                

                if uploaded_image:
                    if st.button("Check"):
                        with st.spinner("Verifying Image..."):
                            is_xray = test.is_hand_xray(image)

                        if not is_xray:
                            st.error("Invalid Image.")
                        
                        else:
                            with st.spinner("Running Fracture Detection..."):
                                output = test.make_prediction(resnet_model, content, conf_threshold)
                                
                                print(output[0])

                                fig, _ax, class_name = test.plot_image_from_output(content[0].detach(), output[0])

                                with col2:
                                    st.image(test.figure_to_array(fig), caption="Result", width=500)
                                    try:
                                        with st.expander("Detection Results"):
                                            if class_name == None:
                                                st.write("No Fracture Detected")
                                            else:
                                                st.write("Fracture Detected")
                                    except Exception as ex:
                                        st.write("No image is uploaded yet!")
                                        st.write(ex)
    else:
        st.write("Please upload an image to test")

    st.markdown("### Do's")
    st.write("-> Only upload human hand X-ray image.")
    st.write("-> Upload high-resolution X-ray images for accurate predictions.")
    st.write("-> Ensure good lighting and contrast in images.")
    st.write("-> Follow correct file format (JPG/PNG).")

    st.markdown("### Dont's")
    st.write("-> Don't upload blurry or distorted images.")
    st.write("-> Avoid using images with excessive noise.")
    st.write("-> Don't refresh the page while processing.")