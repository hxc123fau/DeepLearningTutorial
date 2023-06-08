from pathlib import Path

import numpy as np
import tensorflow as tf

import data
import evaluation

BATCH_SIZE      = 1
EVAL_MEASURES   = ['ClasswiseF1']
CLASSNAMES      = ['crack', 'inactive']
NUM_CLASSES     = len(CLASSNAMES)
DATASET_TEST    = Path('.') / 'data' / 'train.csv'  # Note: We of course don't test on the train set, this is just for you to be able to run the script


class Test:

    def __init__(self, inputs, predictions, ds_test, evaluation):
        self._model_inputs = inputs
        self._predictions = predictions
        self._ds_test = ds_test
        self._evaluation = evaluation

    def run(self, sess):

        for inputs, labels in self._ds_test:
            predictions = sess.run(self._predictions, feed_dict={
                self._model_inputs: inputs,
            })

            if not predictions.shape == labels.shape:
                raise AssertionError("Shape of prediction does not match expected shape.")

            self._evaluation.add_batch(predictions, labels)

        return self._evaluation


if __name__ == '__main__':

    ds = data.Dataset(DATASET_TEST, CLASSNAMES, BATCH_SIZE, False, False)
    ev = evaluation.create_evaluation(EVAL_MEASURES, CLASSNAMES)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 'out')

        predictions = tf.get_default_graph().get_tensor_by_name('model/predictions:0')
        inputs = tf.get_default_graph().get_tensor_by_name('model/inputs:0')
        tester = Test(inputs, predictions, ds, ev)

        # test
        ev = tester.run(sess)

        result = ev._measures[0].values()
        result.append(np.mean(result))
        result = ','.join([str(r) for r in result])
        print(result)

