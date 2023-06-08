
import numpy as np
import tensorflow as tf
import time

start = time.clock()
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

# 加入一些噪声
noise = np.random.normal(0, 0.05, x_data.shape)

y_data = np.square(x_data) - 0.5 + noise
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 定义添加层的函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



# 构造隐藏层

# with tf.device('/cpu:0'):
    h1 = add_layer(xs, 1, 2000, activation_function=tf.nn.relu)

# 构造输出层
    prediction = add_layer(h1, 2000, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化所以变量（注：老版本和新版本的方法名字有差异）
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # if i % 50 == 0:

print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

elapsed = (time.clock() - start)
print("Time used:",elapsed)
