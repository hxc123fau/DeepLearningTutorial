import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

class Trainer:

    def __init__(self, loss, predictions, optimizer, ds_train, ds_validation, stop_patience, evaluation, inputs, labels):
        '''
            Initialize the trainer

            Args:
                loss            an operation that computes the loss
                predictions     an operation that computes the predictions for the current
                optimizer       optimizer to use
                ds_train        instance of Dataset that holds the training data
                ds_validation   instance of Dataset that holds the validation data
                stop_patience   the training stops if the validation loss does not decrease for this number of epochs
                evaluation      instance of Evaluation
                inputs          placeholder for model inputs
                labels          placeholder for model labels
        '''

        self._train_op = optimizer.minimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._ds_train = ds_train
        self._ds_validation = ds_validation
        self._stop_patience = stop_patience
        self._evaluation = evaluation
        self._validation_losses = []
        self._model_inputs = inputs
        self._model_labels = labels
        self.epoche=0

        with tf.variable_scope('model', reuse = True):
            self._model_is_training = tf.get_variable('is_training', dtype = tf.bool)

    def _train_epoch(self, sess):
        '''
            trains for one epoch and prints the mean training loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''
        self._iter_train=iter(self._ds_train)
        self._train_loss=[]

        self.epoche+=1
        print('self.epoche',self.epoche)
        iterator=self._ds_train.__iter__()
        max_one_batch_date = int(self._ds_train._len / self._ds_train._batch_size)
        # print('one_batch_date',one_batch_date)
        small_batch_date=150
        # print('batch_size',batch_size)
        sum_loss=0

        for i in range(small_batch_date):
            batch_image, batch_label = iterator.__next__()

            sess.run(self._train_op,
                     feed_dict={self._model_inputs: batch_image, self._model_labels: batch_label,
                                self._model_is_training: True})

            loss, prediction = sess.run([self._loss, self._predictions],
                                        feed_dict={self._model_inputs: batch_image, self._model_labels: batch_label,
                                                   self._model_is_training: True})
            sum_loss+=loss

        average_loss=sum_loss/small_batch_date
        print('train_average_loss',average_loss)

        # TODO

        pass

    def _valid_step(self, sess):
        '''
            run the validation and print evalution + mean validation loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''

        # TODO
        self._iter_train=iter(self._ds_train)
        self._train_loss=[]

        iterator = self._ds_train.__iter__()
        max_one_batch_date=int(self._ds_train._len/self._ds_train._batch_size)
        small_batch_date=20
        sum_loss = 0

        for i in range(small_batch_date):
            batch_image, batch_label = iterator.__next__()

            loss, prediction = sess.run([self._loss, self._predictions],
                                        feed_dict={self._model_inputs: batch_image, self._model_labels: batch_label,
                                                   self._model_is_training: False})
            # self._validation_losses.append(loss)
            sum_loss+=loss
            self._evaluation.add_batch(prediction, batch_label)

        average_loss = sum_loss / small_batch_date
        self._validation_losses.append(average_loss)
        print('test_average_loss', average_loss)
        self._evaluation.flush()

        pass

    def _should_stop(self):
        '''
            determine if training should stop according to stop_patience
        '''

        # TODO
        if self._validation_losses.__len__() < self._stop_patience:
            return False
            # lowest_loss=self._validation_losses[-1]
            # if lowest_loss==min(self._validation_losses):
            #     return False
            # else:
            #     return True

        else:
            return True

        pass

    def run(self, sess, num_epochs = -1):
        '''
            run the training until num_epochs exceeds or the validation loss did not decrease
            for stop_patience epochs

            args:
                sess        the tensorflow session that should be used
                num_epochs  limit to the number of epochs, -1 means not limit
        '''

        # initial validation step
        self._valid_step(sess)

        i = 0

        # training loop
        while i < num_epochs or num_epochs == -1:
        # while i < 10:
            self._train_epoch(sess)
            self._valid_step(sess)
            i += 1

            if self._should_stop():
                break

