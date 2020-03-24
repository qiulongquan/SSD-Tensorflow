#encoding:utf8
"""
这个是tensorflow中断后继续上次的训练的程序，采用的是随机获取值然后训练的方法，
这个方法是有问题的，有可能有些样品永远都没有被训练，而有些样品被训练了多次。
但是程序本身实现了继续训练的功能。可以参考使用。
"""
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_data(noise=0.1):
    from sklearn.datasets import make_moons
    m = 2000
    X_moons, y_moons = make_moons(m, noise=noise, random_state=42)
    print(X_moons.shape)
    print(X_moons[0])
    print(y_moons.shape)
    print(y_moons[0])
    return X_moons, y_moons


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


def fc_layers(input_tensor,regularizer):
    HINDENN1 = 6
    HINDENN2 = 4
    print("input_tensor.shape={}".format(input_tensor.shape))
    with tf.name_scope("full-connect-layer"):
        fc1 = tf.layers.dense(input_tensor, HINDENN1, activation=tf.nn.elu, \
            kernel_regularizer=regularizer, name="fc1")
        fc2 = tf.layers.dense(fc1, HINDENN2, activation=tf.nn.elu, \
            kernel_regularizer=regularizer, name="fc2")
    return fc2


def train(data, label,learning_rate,lambd,n_epochs,batch_size):
    test_ratio = 0.2
    test_size = int(len(data) * test_ratio)
    X_train = data[:-test_size]
    print(X_train.shape)
    print(X_train[0])
    X_test = data[-test_size:]
    y_train = label[:-test_size]
    y_test = label[-test_size:]

    n_inputs = X_train.shape[1]
    n_outputs = len(set(y_train))
    with tf.name_scope("input"):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

    regularizer = tf.contrib.layers.l2_regularizer(lambd)
    fc2 = fc_layers(X, regularizer)
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc2, n_outputs, kernel_regularizer=regularizer, name="output")
        print("logits.shape={}".format(logits.shape))

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
        loss_summary = tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
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

# utcnow是英国时间，需要变成日本本地时间now()
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = './logs/' + now
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver()

    n_epochs = n_epochs
    batch_size = batch_size
    n_batches = int(np.ceil(len(data) / batch_size))
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
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                print(X_batch.shape)
                print(y_batch.shape)
                # sess.run(train_op这个是进行实际的训练
                sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
                # sess.run([loss, summary_op, predictions, accuracy]这个是获取训练的summary，保存到logs里面，然后使用tensorboard来查看结果
            loss_val, summary_str, test_pred, test_acc = sess.run(
                                            [loss, summary_op, predictions, accuracy], \
                                            feed_dict={X: X_test, y: y_test})

            file_writer.add_summary(summary_str, epoch)
            if epoch % 50 == 0:
                print("Epoch:", epoch, "\tLoss:", loss_val, "\tAcc:", test_acc)
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))

        saver.save(sess, final_model_path)
        y_pred = predictions.eval(feed_dict={X: X_test, y: y_test})
        print('precision_score', precision_score(y_test, y_pred))
        print('recall_score', recall_score(y_test, y_pred))

        sess.close()


if __name__ == '__main__':
    X_moons, y_moons = load_data(noise=0.1)

    learning_rate = 0.001
    lambd = 0.01
    n_epochs = 2000
    batch_size = 64
    train(X_moons, y_moons, learning_rate, lambd, n_epochs, batch_size)
