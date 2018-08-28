import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from sagemaker_tensorflow import PipeModeDataset
from tensorflow.contrib.data import map_and_batch

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'
PREFETCH_SIZE = 10
BATCH_SIZE = 256
NUM_PARALLEL_BATCHES = 10
MAX_EPOCHS = 20


def _conv_pool(inputs, kernel_shape, kernel_count, padding_type):
    # Convolutional Layer 
    conv = tf.layers.conv2d(
      inputs=inputs,
      filters=kernel_count,
      kernel_size=kernel_shape,
      padding=padding_type,
      activation=tf.nn.relu)

    # Pooling Layer 
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
    return pool

    

def model_fn(features, labels, mode, params):
    learning_rate = params.get("learning_rate", 0.0001)
    dropout_rate = params.get("dropout_rate", 0.8)
    nw_depth = params.get("nw_depth", 2)
    optimizer_type = params.get("optimizer_type", 'adam')

    # Input Layer
    X = tf.reshape(features[INPUT_TENSOR_NAME], [-1, 28, 28, 1])
    
    # Series of convolutional layers
    for i in range(nw_depth):
        X = _conv_pool(X, [5,5], 2^(5+i), 'same')
    
    # Dense Layer
    X_flat = tf.layers.flatten(X)
    dense = tf.layers.dense(inputs=X_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10) # default activation is linear combination

    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)


    if mode == tf.estimator.ModeKeys.TRAIN:
        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _input_fn(channel):
    """Returns a Dataset which reads from a SageMaker PipeMode channel."""
    features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64) 
        }

    def parse(record):
        parsed = tf.parse_single_example(record, features)
        
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image.set_shape([784])
        image = tf.cast(image, tf.float32) * (1. / 255)
        label = tf.cast(parsed['label'], tf.int32)
        return ({INPUT_TENSOR_NAME: image}, label)

    ds = PipeModeDataset(channel=channel, record_format='TFRecord')

    ds = ds.repeat(MAX_EPOCHS)
    ds = ds.prefetch(PREFETCH_SIZE)
    ds = ds.map(parse, num_parallel_calls=NUM_PARALLEL_BATCHES)
    ds = ds.batch(BATCH_SIZE)
    
    return ds

def train_input_fn(training_dir, params):
    """Returns input function that feeds the model during training"""
    return _input_fn('train')

def eval_input_fn(training_dir, params):
    """Returns input function that feeds the model during evaluation"""
    return _input_fn('eval')


