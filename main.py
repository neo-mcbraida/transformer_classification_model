import decoder
import encoder
import MHA
import transformer
import tensorflow as tf
import random
import numpy as np
from os import listdir
from PIL import Image

d_model = 10

learning_rate = transformer.CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


# @tf.function()
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = transformer_net(inputs, targets, True)
        loss = loss_function(targets, predictions)
    gradients = tape.gradient(loss, transformer_net.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, transformer_net.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(targets, predictions))


def train(batch_size, epochs, inputs, targets):
    num_batches = len(targets) // batch_size
    data = list(zip(inputs, targets))
    for epoch in range(epochs):
        random.shuffle(data)
        for batch in range(num_batches):
            base = batch * num_batches
            inputs, targets = zip(*(data[base:base+batch_size]))
            inputs = np.array(inputs)
            targets = np.array(targets)
            train_step(inputs, targets)
        if batch % 50 == 0:
            print("epoch: {}, batch: {}, loss: {}, accuracy: {}".format(
                epoch, batch, train_loss.result(), train_accuracy.result()))


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

dimensions = 1875  # 25 x 25 x 3 images, flattens to 1875
out_dims = 10  # 10 output categories
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer_net = transformer.Transformer(
    dimensions, 5, dimensions, 5, 1, out_dims)


im_rows = 25
im_cols = 25

path = "G:/AI DataSets/animals_10/raw-img"


def create_data(path):
    images = []
    targets = []
    categories = {
        "butterfly": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "cat": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "chicken": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "cow": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "dog": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "elephant": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "horse": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "sheep": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "spider": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "squirrel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    for file in listdir(path):
        target = categories[file]
        subpath = path + "/" + file
        for img_name in listdir(subpath):
            im_path = subpath + "/" + img_name
            # some png files open with RGBA (including alpha channel)
            img = Image.open(im_path).convert('RGB')
            rgb_im = img.resize((im_rows, im_cols), Image.ANTIALIAS)
            rgb_im = np.array(rgb_im)
            rgb_im = rgb_im/np.array(255)  # normalize
            images.append(rgb_im)
            img.close()
            targets.append(target)
    return images, targets


def save_arrays(inputs, targets):
    np.save('inputs', inputs)
    np.save('targets', targets)
    print("saved data to json files")


def load_arrays():
    return np.load('inputs.npy'), np.load('targets.npy')


def main():
    print("runnning main")
    inputs, targets = create_data(path)
    save_arrays(inputs, targets)
    # inputs, targets = load_arrays()
    train(1, 100, inputs, targets)


if __name__ == "__main__":
    main()
