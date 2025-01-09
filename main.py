import tensorflow as tf
import numpy as np

data_dir = 'TestData'
imagesize = 28

def main():
    #Load in numpy files
    images = np.load("KMNIST/kmnist-test-imgs.npz")
    labels = np.load("KMNIST/kmnist-test-labels.npz")

    images = images['arr_0']
    images = images.astype(np.float32) / 255.0

    labels = labels['arr_0']
    labels = labels.astype(np.int32)
    labels = tf.keras.utils.to_categorical(labels, num_classes=10)

    model = build_model()
    model.fit(images, labels, epochs=5, batch_size=64)



    while(True):
        label_names = [str(labels) for label in labels]
        #Use new image to test prediction
        img_path = input("Give Image Name: ")

        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(imagesize, imagesize))

        img_path = tf.keras.preprocessing.image.img_to_array(img)
        img_path = np.expand_dims(img_path, axis=0)
        #Convert given image to greyscale
        img_path = 0.2989 * img_path[..., 0] + 0.5870 * img_path[..., 1] + 0.1140 * img_path[..., 2]
        img_path = img_path / np.max(img_path)/ 255.0

        predictions = model.predict(img_path)
        predicted_label = np.argmax(predictions)

        print(predicted_label)

#This is specifically for directories that are read in. Leave for now

# def prepare_training():
#     # Training data setup
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#         tf_data,
#         image_size=(imagesize, imagesize),  # resize all photos
#         validation_split=0.2,  # percentage to be training data (80%)
#         subset="training",  # type
#         batch_size=32,
#         label_mode='int',
#         shuffle=True,
#         seed=42
#     )
#     return train_ds
#
# def prepare_validation():
#     # Validation data setup
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         tf_data,
#         image_size=(imagesize, imagesize),
#         validation_split=0.2,
#         subset='validation',
#         batch_size=32,
#         seed=42,
#     )
#     return val_ds

def build_model():
    # Build the Model
    model = tf.keras.models.Sequential([
        # normalize pixels for learning (mnist images are 28/28 grayscale)
        tf.keras.layers.Rescaling(1. / 255, input_shape=(imagesize, imagesize, 1)),

        # Convolutional Layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # Flatten the output for the connected layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),

        # Output layer
        tf.keras.layers.Dense(10, activation='softmax') #10 classes
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



main()