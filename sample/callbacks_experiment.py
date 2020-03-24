import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
# path = f"{getcwd()}/../tmp2/mnist.npz"

# 训练的时候都要加上[gpu_options.allow_growth]选项，可以大幅度降低显存使用量
# 它允许动态的增加显存使用量,而不是一开始就全部占用
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class Mycallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if logs.get('acc')>0.998:
                self.model.stop_training = True
                print("Reached 99.8% accuracy so cancelling training!")
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # YOUR CODE STARTS HERE
    training_images=training_images.reshape(60000,28,28,1)
    training_images=training_images/255
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='tanh'),
        tf.keras.layers.Dense(10,activation='softmax')
            # YOUR CODE ENDS HERE
    ])

    callbacks=Mycallback()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels,epochs=20,callbacks=[callbacks]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]


if __name__=="__main__":
    _, _ = train_mnist_conv()
