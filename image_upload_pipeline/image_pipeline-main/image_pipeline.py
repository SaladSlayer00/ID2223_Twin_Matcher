import boto3
import os
from datasets import load_dataset, load_metric, Image, DatasetDict, Dataset, load_from_disk, concatenate_datasets
from huggingface_hub import HfApi, Repository, HfFolder
import shutil
import random
from PIL import Image

hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if hf_token:
    HfFolder.save_token(hf_token)  # This will save the token for later use by Hugging Face libraries
else:
    raise ValueError("Hugging Face token not found. Make sure it is passed as an environment variable.")

def extract_class_name_from_folder(key):
    class_name = key.split('/')[0]
    return class_name

def download_images(bucket_name):
    s3 = boto3.client('s3')
    print(f"Connecting to bucket: {bucket_name}")
    response = s3.list_objects_v2(Bucket=bucket_name)

    if 'Contents' in response:
        print(f"Found {len(response['Contents'])} objects in bucket.")
        for obj in response['Contents']:
            key = obj['Key']
            class_name = extract_class_name_from_folder(key)
            print(f"Checking object: {key}")
            if key.endswith('.jpg') or key.endswith('.png'):  # Add more image formats if needed
                # Extract folder name and create directory if it doesn't exist
                directory = os.path.dirname(key)
                if directory and not os.path.exists(directory):
                    print(f"Creating directory: {directory}")
                    os.makedirs(directory)

                # Download the file
                print(f"Downloading image to {key}")
                s3.download_file(bucket_name, key, key)
    else:
        print("No contents found in bucket.")
        
    return directory, class_name

local_directory, class_name = download_images('newimagesupload00')

def split_train_test(image_dir, train_ratio=0.9):
    # Creating paths for train and test directories within image_dir
    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    random.shuffle(all_images)

    split_index = int(len(all_images) * train_ratio)

    train_images = all_images[:split_index]
    test_images = all_images[split_index:]

    for image in train_images:
        shutil.move(os.path.join(image_dir, image), os.path.join(train_dir, image))
    
    for image in test_images:
        shutil.move(os.path.join(image_dir, image), os.path.join(test_dir, image))

    return train_dir, test_dir

train_dir, test_dir = split_train_test(local_directory)

def load_image(image_path):
    with Image.open(image_path) as img:
        return img.convert('RGB')  # Ensuring all images are in RGB format


def update_dataset(train_dir, test_dir, class_name):
    dataset = load_dataset("SaladSlayer00/twin_matcher_data")

    new_train_data = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(train_dir, filename)
            image = load_image(image_path)
            new_train_data.append({'image': image, 'label': class_name})

    new_train_dataset = Dataset.from_dict({'image': [item['image'] for item in new_train_data], 
                                           'label': [item['label'] for item in new_train_data]},
                                           features=dataset['train'].features)
    updated_train_dataset = concatenate_datasets([dataset['train'], new_train_dataset])

    new_test_data = []
    for filename in os.listdir(test_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(test_dir, filename)
            image = load_image(image_path)
            new_test_data.append({'image': image, 'label': class_name})

    new_test_dataset = Dataset.from_dict({'image': [item['image'] for item in new_test_data], 
                                          'label': [item['label'] for item in new_test_data]},
                                          features=dataset['test'].features)
    updated_test_dataset = concatenate_datasets([dataset['test'], new_test_dataset])

    updated_dataset_dict = DatasetDict({
        'train': updated_train_dataset,
        'test': updated_test_dataset
    })

    updated_dataset_dict.push_to_hub('SaladSlayer00/twin_matcher_data')

update_dataset(train_dir, test_dir, class_name)
