import joblib.numpy_pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from keras_nlp.layers import PositionEmbedding
from tqdm import tqdm
import matplotlib.pyplot as plt
l = tf.keras.layers
import zipfile as zf


tf_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
session = tf.compat.v1.Session(config=tf_config)
tf.compat.v1.keras.backend.set_session(session)

dtype = "mixed_float16"
policy = tf.keras.mixed_precision.Policy(dtype)
tf.keras.mixed_precision.set_global_policy(policy)
dtype = "float16" if dtype == "mixed_float16" else "float32"


ZIP_FILE_PATH = "Fashion-MNIST.zip"


def load_mnist_fashion_dataset():
    global ZIP_FILE_PATH
    #reading zip file and extracting the dataset
    with zf.ZipFile(ZIP_FILE_PATH) as zip_file:
        with zip_file.open("fashion-mnist_test.csv") as f:
            unprocessed_test_data = np.asarray(pd.read_csv(f))
        with zip_file.open("fashion-mnist_train.csv") as f:
            unprocessed_train_data = np.asarray(pd.read_csv(f))

    #list-comprehension for input_data
    x_train_data = np.asarray([np.reshape(arr[1:], (28, 28, 1)) for arr in unprocessed_train_data])
    x_test_data = np.asarray([np.reshape(arr[1:], (28, 28, 1)) for arr in unprocessed_test_data])

    y_train_data = []
    y_test_data = []

    #converting the output class-labels to one-hot-encoded arrays
    for i in range(len(unprocessed_train_data)):
        class_target = unprocessed_train_data[i][0]
        y = np.zeros((10,), dtype=dtype)
        y[class_target] = 1
        y_train_data.append(y)
    y_train_data = np.asarray(y_train_data)

    for i in range(len(unprocessed_test_data)):
        class_target = unprocessed_test_data[i][0]
        y = np.zeros((10,), dtype=dtype)
        y[class_target] = 1
        y_test_data.append(y)
    y_test_data = np.asarray(y_test_data)

    return x_train_data, y_train_data, x_test_data, y_test_data


x_train, y_train, x_test, y_test = load_mnist_fashion_dataset()

#setting the Hyper-Parameters as well as crucial other variables
#necessary for the performance comparison later
EPOCHS = 100
BATCH_SIZES = [64, 256, 1024, 4096]
LEARNING_RATES = [1e-2, 1e-3, 5e-4, 1e-4]
MODELS = []
MODEL_NAMES = []
HISTORIES = []

#creation of 1st model using CNN's and only a Dense layer as the output layer
cnn_model = tf.keras.Sequential([l.Input((28, 28, 1)),
                                 l.Rescaling(1./127.5, -1.),
                                 l.MaxPooling2D(2),
                                 l.Conv2D(16, 3, 1, 'same',
                                          kernel_initializer='glorot_normal', activation='tanh'),
                                 l.MaxPooling2D(2),
                                 l.Conv2D(32, 3, 1, 'same',
                                          kernel_initializer='glorot_normal', activation='tanh'),
                                 l.Conv2D(32, 3, 1, 'same',
                                          kernel_initializer='glorot_normal', activation='tanh'),
                                 l.MaxPooling2D(2),
                                 l.Conv2D(64, 3, 1, 'same',
                                          kernel_initializer='glorot_normal', activation='tanh'),
                                 l.Conv2D(64, 3, 1, 'same',
                                          kernel_initializer='glorot_normal', activation='tanh'),
                                 l.Conv2D(64, 3, 1, 'same',
                                          kernel_initializer='glorot_normal', activation='tanh'),
                                 l.Flatten(),
                                 l.Dense(10, kernel_initializer='glorot_normal', activation='softmax')
                                 ], name="CNN-Model")
#cnn_model.summary()
MODELS.append(cnn_model)
MODEL_NAMES.append(cnn_model.name)


#creation of 2nd model using patch-extraction, with a CNN for a learnable patch-extractor instead of a fixed one,
#which is standard in most ViT- and Transformer-Models and also only a Dense layer for the output layer
transformer_model_input = l.Input((28, 28, 1))
scaled = l.Rescaling(1./127.5, -1.)(transformer_model_input)

patches = l.Conv2D(64, 4, 4, 'same',
                   kernel_initializer='glorot_normal', activation='tanh')(scaled)
patches = l.Reshape((-1, patches.shape[-1]))(patches)
position = PositionEmbedding(patches.shape[-2])(patches)
x = l.Add()([patches, position])

num_heads = 4
dims = patches.shape[-1]//num_heads

for _ in range(4):
    skip = x
    attention = l.MultiHeadAttention(num_heads, dims, dims)(x, x, x)
    x = l.Add()([attention, skip])

x = l.Flatten()(x)
x = l.Dropout(.1)(x)
transformer_model_output = l.Dense(10, kernel_initializer='glorot_normal', activation='softmax')(x)

transformer_model = tf.keras.Model(inputs=transformer_model_input, outputs=transformer_model_output,
                                   name="Transformer-Model")
#transformer_model.summary()

MODELS.append(transformer_model)
MODEL_NAMES.append(transformer_model.name)


#creation of 3rd model using only Dense layers
dense_model = tf.keras.Sequential([l.Input((28, 28, 1)),
                                   l.Rescaling(1./127.5, -1.),
                                   l.Flatten(),
                                   l.Dropout(.5),
                                   l.Dense(256, kernel_initializer='glorot_normal', activation='tanh'),
                                   l.Dense(10, kernel_initializer='glorot_normal', activation='softmax')
                                   ], name="Dense-Model")

#dense_model.summary()
MODELS.append(dense_model)
MODEL_NAMES.append(dense_model.name)


#for-loop for training models with all different hyperparameter-configurations
for model in tqdm(MODELS):
    for batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            #dynamic dataset-creation due to differing hyperparameters
            TRAINING_DATASET = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), tf.constant(y_train)))
            TRAINING_DATASET = TRAINING_DATASET.batch(batch_size=batch_size, deterministic=False,
                                                      drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
            TRAINING_DATASET = TRAINING_DATASET.shuffle(len(x_train), reshuffle_each_iteration=True)

            TEST_DATASET = tf.data.Dataset.from_tensor_slices((tf.constant(x_test), tf.constant(y_test)))
            TEST_DATASET = TEST_DATASET.batch(batch_size=batch_size, deterministic=False,
                                              drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
            TEST_DATASET = TEST_DATASET.shuffle(len(x_test), reshuffle_each_iteration=True)

            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
            #creating a clone of the model, so that the model does not use the already pre-trained model-parameters
            #from previous loop-iterations with the previous hyperparameters
            temp_model = tf.keras.models.clone_model(model)
            temp_model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics='accuracy')
            print(f"\n\n{model.name}, lr={learning_rate}, batch_size={batch_size}\n")
            history = temp_model.fit(TRAINING_DATASET, epochs=EPOCHS, validation_data=TEST_DATASET)
            #appending all results of the training with all combinations of the hyperparameters
            #for later comparison and visualization
            HISTORIES.append({"model": model.name, "history": history,
                              "learning_rate": learning_rate, "batch_size": batch_size})


dpi = 720
width = 3840*2
height = 2160*2

#plotting the results in graphs and grouping them together into certain categories to keep a better
#readability of the graphs as well as making more fine-detailed comparisons easier whilst still maintaining
#the ability to compare a vast amount of data
plot_directory = 'plots/loss'
os.makedirs(plot_directory, exist_ok=True)
for model_name in MODEL_NAMES:
    for batch_size in BATCH_SIZES:
        temp_plot_directory = os.path.join(plot_directory, model_name, str(batch_size))
        os.makedirs(temp_plot_directory, exist_ok=True)
        plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)
        plt.title(f"{model_name}, Batch-Size: {batch_size}")
        for history in HISTORIES:
            if history["model"] == model_name and history["batch_size"] == batch_size:
                plt.plot(history["history"].history['loss'],
                         label=f'lr={history["learning_rate"]}- TRAIN')
                plt.plot(history["history"].history['val_loss'],
                         label=f'lr={history["learning_rate"]}- TEST')
            else:
                continue
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=1.)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}-Train-And-Test-Loss-Plot.png'))
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}-Train-And-Test-Loss-Vector-Plot.svg'))
        plt.close()

    for learning_rate in LEARNING_RATES:
        temp_plot_directory = os.path.join(plot_directory, model_name, str(learning_rate))
        os.makedirs(temp_plot_directory, exist_ok=True)
        plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)
        plt.title(f"{model_name}, Learning-Rate: {learning_rate}")
        for history in HISTORIES:
            if history["model"] == model_name and history["learning_rate"] == learning_rate:
                plt.plot(history["history"].history['loss'],
                         label=f'batch-size={history["batch_size"]}- TRAIN')
                plt.plot(history["history"].history['val_loss'],
                         label=f'batch-size={history["batch_size"]}- TEST')
            else:
                continue
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=1.)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plot_directory, f'Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Plot.png'))
        plt.savefig(os.path.join(temp_plot_directory, f'Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Vector-Plot.svg'))
        plt.close()

for learning_rate in LEARNING_RATES:
    for batch_size in BATCH_SIZES:
        temp_plot_directory = os.path.join(plot_directory, "model_comparison", "LR " +
                                           str(learning_rate) + "_" + "BS " + str(batch_size))
        os.makedirs(temp_plot_directory, exist_ok=True)
        plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)
        plt.title(f"Batch-Size: {batch_size}, Learning-Rate: {learning_rate}")
        for history in HISTORIES:
            if history["batch_size"] == batch_size and history["learning_rate"] == learning_rate:
                plt.plot(history["history"].history['loss'],
                         label=f'{history["model"]} - TRAIN')
                plt.plot(history["history"].history['val_loss'],
                         label=f'{history["model"]} - TEST')
            else:
                continue

        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=1.)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}, Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Plot.png'))
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}, Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Vector-Plot.svg'))
        plt.close()


plot_directory = 'plots/accuracy'
os.makedirs(plot_directory, exist_ok=True)
for model_name in MODEL_NAMES:
    for batch_size in BATCH_SIZES:
        temp_plot_directory = os.path.join(plot_directory, model_name, str(batch_size))
        os.makedirs(temp_plot_directory, exist_ok=True)
        plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)
        plt.title(f"{model_name}, Batch-Size: {batch_size}")
        for history in HISTORIES:
            if history["model"] == model_name and history["batch_size"] == batch_size:
                plt.plot(history["history"].history['accuracy'],
                         label=f'lr={history["learning_rate"]}- TRAIN')
                plt.plot(history["history"].history['val_accuracy'],
                         label=f'lr={history["learning_rate"]}- TEST')
            else:
                continue
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=1.)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}-Train-And-Test-Loss-Plot.png'))
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}-Train-And-Test-Loss-Vector-Plot.svg'))
        plt.close()

    for learning_rate in LEARNING_RATES:
        temp_plot_directory = os.path.join(plot_directory, model_name, str(learning_rate))
        os.makedirs(temp_plot_directory, exist_ok=True)
        plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)
        plt.title(f"{model_name}, Learning-Rate: {learning_rate}")
        for history in HISTORIES:
            if history["model"] == model_name and history["learning_rate"] == learning_rate:
                plt.plot(history["history"].history['accuracy'],
                         label=f'batch-size={history["batch_size"]}- TRAIN')
                plt.plot(history["history"].history['val_accuracy'],
                         label=f'batch-size={history["batch_size"]}- TEST')
            else:
                continue
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=1.)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plot_directory, f'Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Plot.png'))
        plt.savefig(os.path.join(temp_plot_directory, f'Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Vector-Plot.svg'))
        plt.close()

for learning_rate in LEARNING_RATES:
    for batch_size in BATCH_SIZES:
        temp_plot_directory = os.path.join(plot_directory, "model_comparison", "LR " +
                                           str(learning_rate) + " " + "BS " + str(batch_size))
        os.makedirs(temp_plot_directory, exist_ok=True)
        plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)
        plt.title(f"Batch-Size: {batch_size}, Learning-Rate: {learning_rate}")
        for history in HISTORIES:
            if history["batch_size"] == batch_size and history["learning_rate"] == learning_rate:
                plt.plot(history["history"].history['accuracy'],
                         label=f'{history["model"]} - TRAIN')
                plt.plot(history["history"].history['val_accuracy'],
                         label=f'{history["model"]} - TEST')
            else:
                continue

        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=1.)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}, Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Plot.png'))
        plt.savefig(os.path.join(temp_plot_directory, f'Batch-Size {batch_size}, Learning-Rate {learning_rate}'
                                                      f'-Train-And-Test-Loss-Vector-Plot.svg'))
        plt.close()