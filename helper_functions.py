import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir
    )
    print(f"TensorBoard log 파일들은 {log_dir}에 저장했습니다.")
    return tensorboard_callback

def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # Plot Loss
    plt.plot(epochs, loss, label = "traning_loss")
    plt.plot(epochs, val_loss, label = "val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label = "traning_accuracy")
    plt.plot(epochs, val_accuracy, label = "val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


import os

def walk_through_dir(directory_name):
    for dirpath, dirnames, filenames in os.walk(directory_name):
        print(f"{dirpath} 디렉토리에는 {len(dirnames)}개의 디렉토리가 존재하고 {len(filenames)}개의 파일이 존재합니다.")


import zipfile

def unzip(filename):
    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall()
    zip_ref.close()