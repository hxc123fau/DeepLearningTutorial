import tensorflow as tf
import numpy as np
from tensorflow import keras


def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
            dropout         dropout rate during training
    '''
    
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)

    # TODO
    input_tensor=x
    dropout_rate = 0.5
    channels=[96,192,384,256,256]
    filter_size=[11,5,3,3,3]
    stride=[4,1,1,1,1]

    fcl_weights = tf.contrib.layers.xavier_initializer()
    conv_weights = tf.contrib.layers.xavier_initializer()
    Lambda=0.0055

    out=conv_bn_re(input_tensor,channels[0],filter_size[0],stride[0],conv_weights,Lambda)
    out=maxpool(out,3,2)
    out=conv_bn_re(out,channels[1],filter_size[1],stride[1],conv_weights,Lambda)
    out = maxpool(out, 3, 2)
    out = conv_bn_re(out, channels[2], filter_size[2], stride[2],conv_weights,Lambda)
    out = conv_bn_re(out, channels[3], filter_size[3], stride[3],conv_weights,Lambda)
    out = conv_bn_re(out, channels[4], filter_size[4], stride[4],conv_weights,Lambda)
    out = maxpool(out, 3, 2)
    out=dropout(out,dropout_rate)
    out=fcl(out,4096,fcl_weights,Lambda)
    out=relu(out)
    out = dropout(out, dropout_rate)
    out=fcl(out,4096,fcl_weights,Lambda)
    out=relu(out)
    out=fcl(out,num_outputs,fcl_weights,Lambda)
    return out

def conv_bn_re(input_layer,channels,filter_size,stride,conv_weights,Lambda):
    out_conv=conv(input_layer, channels, filter_size, stride,conv_weights,Lambda)
    out_bn=bn(out_conv,Lambda)
    out_relu=relu(out_bn)
    return out_relu


def conv(input_layer, number_filters,filters_size,CONV_STRIDE_SIZE,conv_weights,Lambda):
    conv1_layer = tf.layers.conv2d(input_layer, filters=number_filters, kernel_size=(filters_size, filters_size),
                                   strides=(CONV_STRIDE_SIZE, CONV_STRIDE_SIZE), padding='same', activation=None,use_bias=True,
                                   kernel_initializer=conv_weights, kernel_regularizer=keras.regularizers.l2(Lambda))
    return conv1_layer

def bn(x,Lambda):
    # x = utils._bn(x, self.is_train, self._global_step, name)
    x_norm = tf.layers.batch_normalization(x,gamma_regularizer=keras.regularizers.l2(Lambda))
    return x_norm

def fcl(input_layer,n_weights_fc1,fcl_weights,Lambda):
    input_layer = tf.layers.flatten(input_layer)
    # Then we can multiply the input_layer with the weights matrix
    # and add a bias in a fully connected layer (tf: dense layer ) # units = dimensionality of the output space.
    fc = tf.layers.dense(input_layer, units=n_weights_fc1, kernel_initializer=fcl_weights,
                         kernel_regularizer=keras.regularizers.l2(Lambda), activation=None, use_bias=True)
    return fc

def relu(pre_activation):
    layer_output = tf.nn.relu(pre_activation)
    return layer_output

def maxpool(input_layer,POOL_KERNEL_SIZE,POOL_STRIDE_SIZE):
    pooled_output = tf.layers.max_pooling2d(input_layer, pool_size=(POOL_KERNEL_SIZE, POOL_KERNEL_SIZE),
                                            strides=(POOL_STRIDE_SIZE, POOL_STRIDE_SIZE), padding='valid')
    return pooled_output

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)