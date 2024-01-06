
# Twin Matcher - Find Your Lookalike 🌟

Welcome to Twin Matcher, your gateway to discovering your celebrity doppelgänger! 🔍 This is the final project for the **ID2223 - Scalable Machine Learning & Deep Learning** course at KTH. Notebooks and files are commented, to ensure the best accessibility.

## Overview 🚀

Twin Matcher leverages the power of the [Microsoft ResNet-50](https://huggingface.co/microsoft/resnet-50) model to find your lookalike. The system features a seamless pipeline that includes image preprocessing, data handling, model training, and inference. It allows you to visualize your prediction and upload your own data to become part of the dataset and find your long-lost twin. The system is designed for efficiency, minimizing the time from data acquisition to model improvement. By leveraging continuous integration practices, the pipeline is both robust and capable of handling large volumes of data without sacrificing performance. We ensure to get coherent batch updates and the best possible matches with continuous model retraining.

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
**Objective**: Transform original dataset images into a refined format suitable for machine learning models. The notebook preprocesses the original dataset, [lansinuote/simple_facenet🤗](https://huggingface.co/datasets/lansinuote/simple_facenet), and makes it fit for training image classification networks. It solves class imbalance, checks for label coherence, and uploads the new dataset on the [Hugging Face Platform🤗](https://huggingface.co/datasets/SaladSlayer00/twin_matcher_data), ensuring it keeps the right format. This allows the data to benefit from the functionalities offered by the datasets library, and easy retrieval thanks to full integration with the rest of the environment. The image dataset is composed of a DataDict of images (which are [160 x 160] cuts of faces of various celebrities) and labels (the associated celebrity names). There's a total of 105 initial identities, each presenting 85 images in the training split. 
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
