# https://tuaiznblqcvchvaqbzxmjp.coursera-apps.org/notebooks/week2/Exercise2-Question-Copy1.ipynb
# 参考上面的例子，callbacks 提前退出 训练的基本代码

# 【秒速で無料GPUを使う】深層学習実践Tips on Colaboratory
# https://qiita.com/tomo_makes/items/b3c60b10f7b25a0a5935

import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # y_train = y_train / 255.0
    # y_test = y_test / 255.0
    return x_train, y_train, x_test, y_test


def fc_layers(input_tensor,regularizer):
    HINDENN1 = 512
    print("input_tensor.shape={}".format(input_tensor.shape))
    with tf.name_scope("full-connect-layer"):
        fc1 = tf.layers.flatten(input_tensor, name="fc1")
        fc2 = tf.layers.dense(fc1, HINDENN1, activation=tf.nn.relu, \
            kernel_regularizer=regularizer, name="fc2")
    return fc2


def random_batch(x_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_batch = x_train[rnd_indices]
    # print("x_batch.shape={}".format(x_batch.shape))
    y_batch = y_train[rnd_indices]
    # print("y_batch.shape={}".format(y_batch.shape))
    return x_batch, y_batch


def train(x_train, y_train, x_test, y_test,learning_rate, lambd, epochs, batch_size):
    n_inputs = x_train.shape[1]
    n_outputs = 10

    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=(None, n_inputs, n_inputs), name="x")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

    regularizer = tf.contrib.layers.l2_regularizer(lambd)
    fc2 = fc_layers(x, regularizer)
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc2, n_outputs, activation=tf.nn.softmax, kernel_regularizer=regularizer,
                                 name="output")
        # print("logits.shape={}".format(logits.shape))

    with tf.name_scope('loss'):
        # 损失函数的部分有一些问题，不能降低loss，但是准确率却可以提高，和标准程序基本一样98.5%左右
        # predictions = tf.argmax(logits, 1)
        # y_pred = predictions.eval(feed_dict={x: x_test, y: y_test})
        # xentropy = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
        loss_summary = tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    print("global_step={}".format(global_step))
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.name_scope('eval'):
        predictions = tf.argmax(logits, 1)
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        acc_summary = tf.summary.scalar('acc', accuracy)

    summary_op = tf.summary.merge([loss_summary, acc_summary])

    checkpoint_path = "./chickpoints/model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./chickpoints/model/model_final"
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = './logs/' + now
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver()

    n_epochs = epochs
    batch_size = batch_size
    n_batches = int(np.ceil(len(x_train) / batch_size))
    print("n_batches={}".format(n_batches))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        if os.path.isfile(checkpoint_epoch_path):
            # if the checkpoint file exists, restore the model and load the epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for batch_index in range(n_batches):
                # 随机取一个batch的值，付给X_batch, y_batch
                x_batch, y_batch = random_batch(x_train, y_train, batch_size)
                # sess.run(train_op这个是进行实际的训练
                sess.run(train_op, feed_dict={x: x_batch, y: y_batch})
                # sess.run([loss, summary_op, predictions, accuracy]这个是获取训练的summary，保存到logs里面，然后使用tensorboard来查看结果

            train_loss_val, train_summary_str, train_pred, train_acc = sess.run(
                [loss, summary_op, predictions, accuracy], \
                feed_dict={x: x_train, y: y_train})

            loss_val, summary_str, test_pred, test_acc = sess.run(
                [loss, summary_op, predictions, accuracy], \
                feed_dict={x: x_test, y: y_test})

            if (test_acc > 0.9810):
                print("\nReached 98.1% accuracy so cancelling training!")
                break
            else:
                print("\nReached {} accuracy.".format(test_acc))

            file_writer.add_summary(summary_str, epoch)
            print("Epoch:", epoch, "\ttrain_Loss:", train_loss_val, "\ttrain_Acc:", train_acc, "\ttest_Loss:", loss_val, "\ttest_Acc:", test_acc)
            if epoch % 10 == 0:
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))

        saver.save(sess, final_model_path)
        y_pred = predictions.eval(feed_dict={x: x_test, y: y_test})
        print('precision_score', precision_score(y_test, y_pred))
        print('recall_score', recall_score(y_test, y_pred))

        sess.close()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    # 学习率维持在0.001比较好，太小了训练时间会提高很多，太大了抖动会很大。这个需要不同学习率值进行测试。
    learning_rate = 0.001
    lambd = 0.01
    epochs = 1000
    batch_size = 128
    train(x_train, y_train, x_test, y_test, learning_rate, lambd, epochs, batch_size)
