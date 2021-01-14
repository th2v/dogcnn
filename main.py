import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D()
        self.flatten = Flatten()
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d2(x)
        return self.d3(x)


def imageSetLoad():
    data_dir = 'D:/image_dataset/archive/training_set/training_set'
    batch_size = 10
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    data_dir = 'D:/image_dataset/archive/test_set/test_set'

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == "__main__":

    train_ds, test_ds = imageSetLoad()

    model = MyModel()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    EPOCHS = 10

    for epoch in range(EPOCHS):
        for images, labels in train_ds:
            train_step(images / 255., labels)

        for test_images, test_labels in test_ds:
            test_step(test_images / 255., test_labels)

        template = 'epoch: {}, loss: {}, accuracy: {}, test loss: {}, test accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

    tf.keras.models.save_model(
        model, './testweight', overwrite=True, include_optimizer=True, save_format='tf',
        signatures=None, options=None
    )