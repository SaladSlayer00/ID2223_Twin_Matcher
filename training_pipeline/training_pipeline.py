# -*- coding: utf-8 -*-


from transformers.utils import send_example_telemetry
from datasets import load_metric
from datasets import load_dataset
from transformers import AutoFeatureExtractor
import numpy as np
import tensorflow as tf
from keras import backend
import matplotlib.pyplot as plt
from transformers import TFAutoModelForImageClassification
from transformers import AdamWeightDecay
from transformers import DefaultDataCollator
import numpy as np
from transformers.keras_callbacks import KerasMetricCallback
from transformers.keras_callbacks import PushToHubCallback
from tensorflow.keras.callbacks import TensorBoard
import os
from huggingface_hub import HfApi, HfFolder
from tensorflow.keras.callbacks import ModelCheckpoint


dataset = load_dataset("SaladSlayer00/twin_matcher")
send_example_telemetry("image_classification_notebook", framework="tensorflow")
model_checkpoint = "microsoft/resnet-50" # pre-trained model from which to fine-tune
batch_size = 16 # batch size for training and evaluation
metric = load_metric("accuracy")
example = dataset["train"][10]

labels = dataset["train"]["label"]  # Replace "train" with the split you want to access

# Now, you can iterate over the labels
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)


def normalize_img(img, mean, std):
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / tf.maximum(std, backend.epsilon())


def get_resize_shape(img, size):
    if isinstance(size, tuple):
        return size

    height, width, _ = img.shape
    target_height = int(size * height / width) if height > width else size
    target_width = int(size * width / height) if width > height else size
    return (target_height, target_width)


def get_random_crop_size(img, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
    height, width, channels = img.shape
    img_ratio = width / height
    crop_log_ratio = np.random.uniform(*np.log(ratio), size=1)
    crop_ratio = np.exp(crop_log_ratio)
    crop_scale = np.random.uniform(*scale, size=1)

    # Make sure the longest side is within the image size
    if crop_ratio < img_ratio:
        crop_height = int(height * crop_scale)
        crop_width = int(crop_height * crop_ratio)
    else:
        crop_width = int(width * crop_scale)
        crop_height = int(crop_width / crop_ratio)
    return (crop_height, crop_width, channels)


def train_transforms(image, size=200):
    image = tf.keras.utils.img_to_array(image)

    image = tf.image.resize(
        image,
        size=(size, size),
        method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.image.random_flip_left_right(image)
    image /= 255
    image = normalize_img(
        image,
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std
    )
    # All image models take channels first format: BCHW
    image = tf.transpose(image, (2, 0, 1))
    return image


def val_transforms(image, size=200):
    image = tf.keras.utils.img_to_array(image)
    resize_shape = get_resize_shape(image, feature_extractor.size)
    image = tf.image.resize(
        image,
        size=(size,size),
        method=tf.image.ResizeMethod.BILINEAR
    )

    image /= 255
    image = normalize_img(
        image,
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std
    )
    # All image models take channels first format: BCHW
    image = tf.transpose(image, (2, 0, 1))
    return image


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch['pixel_values'] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch['pixel_values'] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch



def unnormalize_img(img, mean, std):
    img = (img * std) + mean
    return img


def process_for_plotting(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = unnormalize_img(
        img=img,
        mean=feature_extractor.image_mean,
        std=feature_extractor.image_std
    )
    img = img * 255
    img = img.astype(int)
    return img


n = 10
fig, ax = plt.subplots(2, n, figsize=(20, 10))


def label_to_int(example):
    example['label'] = label2id[example['label']]
    return example

dataset = dataset.map(label_to_int)

splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
if hf_token:
    HfFolder.save_token(hf_token)  # This will save the token for later use by Hugging Face libraries
else:
    raise ValueError("Hugging Face token not found. Make sure it is passed as an environment variable.")

model = TFAutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)


learning_rate = 5e-5
weight_decay = 0.01
epochs = 10

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)

model.compile(optimizer=optimizer)


data_collator = DefaultDataCollator(return_tensors="np")

train_set = train_ds.to_tf_dataset(
    columns=["pixel_values", "label"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)
val_set = val_ds.to_tf_dataset(
    columns=["pixel_values", "label"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator
)

batch = next(iter(train_set))


# the compute_metrics function takes a Tuple as input:
# first element is the logits of the model as Numpy arrays,
# second element is the ground-truth labels as Numpy arrays.
def compute_metrics(eval_predictions):
    predictions = np.argmax(eval_predictions[0], axis=1)
    metric_val = metric.compute(predictions=predictions, references=eval_predictions[1])
    return {"val_" + k: v for k, v in metric_val.items()}

metric_callback = KerasMetricCallback(
    metric_fn=compute_metrics, eval_dataset=val_set, batch_size=batch_size, label_cols=['labels']
)


tensorboard_callback = TensorBoard(log_dir="./twin_matcher/logs")

model_name = model_checkpoint.split("/")[-1]
push_to_hub_model_id = f"twin_matcher"

push_to_hub_callback = PushToHubCallback(
    output_dir="./twin_matcher",
    hub_model_id=push_to_hub_model_id,
    tokenizer=feature_extractor,
)

callbacks = [metric_callback, tensorboard_callback, push_to_hub_callback]

checkpoint_callback = ModelCheckpoint(
    filepath="./twin_matcher/checkpoints/model-{epoch:04d}.ckpt",
    save_weights_only=True,
    verbose=1,
    save_freq='epoch'
)
callbacks.append(checkpoint_callback)

latest_checkpoint = tf.train.latest_checkpoint("./twin_matcher/checkpoints/")
if latest_checkpoint:
    print(f"Loading from checkpoint: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

print(train_set)
print(val_set)

model.fit(
    train_set,
    validation_data=val_set,
    callbacks=callbacks,
    epochs=35,
    batch_size=batch_size
)


eval_loss = model.evaluate(val_set)

