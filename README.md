
# Twin Matcher - Find Your Celebrity Lookalike 🌟

Welcome to Twin Matcher, your gateway to discovering your celebrity doppelgänger! 🔍 This is the final project for the **ID2223 - Scalable Machine Learning & Deep Learning** course at KTH.

## Overview 🚀

Twin Matcher leverages the power of the [Microsoft ResNet-50](https://huggingface.co/microsoft/resnet-50) model to find your celebrity lookalike. 
<p align="center">
  <img src="twin-matcher-demo.gif" alt="Twin Matcher Demo" width="600">
</p>


## Tools 🛠️

This project is built around a complex system pipeline that ensures efficiency and effectiveness, exploiting the capabilities of different tools and platforms:

- **Microsoft ResNet-50 Model**: Our project utilizes the Microsoft ResNet-50 model, a deep convolutional neural network (CNN) known for its exceptional facial feature extraction capabilities, fitting for this task.

- **Gradio for User Interaction**: Gradio, a user-friendly machine learning interface, connects users with our AI pipeline. It enables seamless image uploads and interactions with our system.

- **Hugging Face Datasets and Models**: Twin Matcher leverages Hugging Face for dataset hosting and model management, promoting collaboration and data sharing within the AI research community.
- **Amazon S3**: The selected Cloud object storage service that offers industry-leading scalability, data availability, and security.
- **GitHub + Github Actions**: Serverless platform for code hosting, versioning, and scheduled executions.
- **Google Colab**: Hoster service for model on-demand training.

A high-level illustration of the system pipeline can be visualized here:


## How It Works 🤖

### 1. Image Preprocessing Notebook 📷

The journey begins with our **Image Preprocessing Notebook**. 📔 This notebook meticulously prepares our celebrity dataset for its starring role in the Twin Matcher project. It transforms raw images into a format that our model can understand and appreciate. 📸

### 2. Data Pipeline 🚢

Our **Data Pipeline** is where the magic happens. 🌟 Users can upload videos through a sleek Gradio app. We capture frames from these videos, identifying faces and associating them with user names. The processed data is then whisked away to an S3 bucket, ready for its next adventure.

### 3. Image Pipeline Python Program 🐍

Periodically, our trusty **Image Pipeline Python Program** sets sail to the S3 bucket to process its contents. 🚀 It carefully extracts facial features and adds them to our project's dynamic dataset, hosted on Hugging Face as a Hugging Face dataset. 🌐 This ensures that our model is always in tune with the latest trends.

### 4. Training Pipeline 🎯

Our fine-tuned Microsoft ResNet-50 model is a star in the making. 🌟 The **Training Pipeline** takes the reins, using a custom program to refine the model's skills. Once it's ready for the spotlight, we upload it to Hugging Face, making it accessible to users far and wide.

## Getting Started 🚀

Want to experience the magic of Twin Matcher? Follow these steps:

1. **Clone the Repository:**


Execute different components of the project, such as the data pipeline, image pipeline, and Gradio apps, as needed.


Acknowledgments 👏
We extend our heartfelt thanks to Hugging Face for providing an incredible platform for hosting datasets and models. 🤗

Enjoy the adventure of finding your celebrity twin with Twin Matcher! ✨
