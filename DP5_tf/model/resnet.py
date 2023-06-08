import tensorflow as tf
import numpy as np
from tensorflow import keras

def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''
    
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    
    # TODO
    # print('image_x',x.shape)
    fcl_weights = tf.contrib.layers.xavier_initializer()
    conv_weights = tf.contrib.layers.xavier_initializer()
    Lambda=0.0057

    out_conv=conv(x,64,7,2,conv_weights,Lambda)
    out_bn=bn(out_conv,Lambda)
    out_relu=relu(out_bn)
    out_pool=tf.layers.max_pooling2d( out_relu , pool_size=( 3 , 3 ), strides =(2,2), padding = 'same' )
    print('out_pool',out_pool.shape)

    number_filters = [64, 128, 256, 512]  # number of filters
    filter_size=3
    strides = [1, 2, 2, 2]
    pool_size=3
    pool_stride=2
    output_size=num_outputs

    out_first_resblock = residual_block_first(out_pool, number_filters[0], filter_size, strides[0],conv_weights,Lambda)
    print('out_first_resblock',out_first_resblock.shape)
    out_resblock = residual_block(out_first_resblock, number_filters[1], filter_size, strides[1],conv_weights,Lambda)
    out_resblock = residual_block(out_resblock, number_filters[2], filter_size, strides[2],conv_weights,Lambda)
    out_resblock = residual_block(out_resblock, number_filters[3], filter_size, strides[3],conv_weights,Lambda)
    # out_global_avg=lobal_avg_pool(out_resblock,pool_size,pool_stride)
    out_global_avg = tf.reduce_mean(out_resblock, axis=[1, 2])
    res=fcl(out_global_avg,output_size,fcl_weights,Lambda)

    return res

def residual_block_first( input_tensor,number_filters,filter_size,strides,conv_weights,Lambda):
    in_channel = input_tensor.get_shape().as_list()[-1]
    input=input_tensor
    output = conv(input,number_filters,filter_size,strides,conv_weights,Lambda)
    output = bn(output,Lambda)
    output = relu(output)
    output = conv(output,number_filters,filter_size,strides,conv_weights,Lambda)
    output = bn(output,Lambda)
    # Merge
    print('input',input.shape,'output',output.shape)
    output = output + input
    output = relu(output)

    return output


def residual_block(input_tensor,number_filters,filter_size,strides,conv_weights,Lambda):
    input=input_tensor
    output = conv(input,number_filters,filter_size,strides,conv_weights,Lambda)
    output = bn(output,Lambda)
    output = relu(output)
    output = conv(output,number_filters,filter_size,1,conv_weights,Lambda)
    output = bn(output,Lambda)
    # Merge
    print('input22',input.shape,'output22',output.shape)
    output = output + conv(input,number_filters,1,2,conv_weights,Lambda)
    output = relu(output)

    return output

def lobal_avg_pool(input_layer, POOL_KERNEL_SIZE, POOL_STRIDE_SIZE):
    pool_output=tf.layers.average_pooling2d(input_layer, POOL_KERNEL_SIZE, POOL_STRIDE_SIZE)
    # pool_output=tf.reduce_mean(input_layer,axis=[1,2],keepdims=False)
    return pool_output

def conv(input_layer, number_filters,filters_size,CONV_STRIDE_SIZE,conv_weights,Lambda):
    conv1_layer = tf.layers.conv2d(input_layer, filters=number_filters, kernel_size=(filters_size, filters_size),
                                   strides=(CONV_STRIDE_SIZE, CONV_STRIDE_SIZE), padding='same', activation=None,use_bias=True,
                                   kernel_initializer=conv_weights, kernel_regularizer=keras.regularizers.l2(Lambda))
    return conv1_layer

def bn(x,Lambda):
    # x = utils._bn(x, self.is_train, self._global_step, name)
    x_norm = tf.layers.batch_normalization(x,beta_regularizer=keras.regularizers.l2(Lambda))
    return x_norm

def fcl(input_layer,n_weights_fc1,fcl_weights,Lambda):
    input_layer = tf.layers.flatten(input_layer)
    # Then we can multiply the input_layer with the weights matrix
    # and add a bias in a fully connected layer (tf: dense layer )# units = dimensionality of the output space.
    fc = tf.layers.dense(input_layer, units=n_weights_fc1, kernel_initializer=fcl_weights,
                         kernel_regularizer=keras.regularizers.l2(Lambda), activation=None, use_bias=True)
    return fc

def relu(pre_activation):
    layer_output = tf.nn.relu(pre_activation)
    return layer_output

def maxpool(input_layer,POOL_KERNEL_SIZE,POOL_STRIDE_SIZE):
    pooled_output = tf.layers.max_pooling2d(input_layer, pool_size=(POOL_KERNEL_SIZE, POOL_KERNEL_SIZE),
                                            strides=(POOL_STRIDE_SIZE, POOL_STRIDE_SIZE), padding='same')
    return pooled_output
