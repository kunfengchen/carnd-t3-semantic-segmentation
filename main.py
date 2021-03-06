import os.path
#import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

helper.maybe_download_pretrained_vgg('./data')

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    default_graph = tf.get_default_graph()
    image_input = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #onebyone = tf.layers.conv2d(vgg_layer7_out, num_classes, 1 , 1)
    #onebyone = tf.layers.conv2d(vgg_layer7_out, 4096, 1 , 1)
    onebyone = helper.conv_1x1(vgg_layer7_out, 4096)
    #onebyone = helper.conv_1x1(vgg_layer7_out, num_classes)
    # trans1 = tf.layers.conv2d_transpose(onebyone, num_classes, 4, strides=(2, 2), name="trans1")
    #trans1 = tf.layers.conv2d_transpose(onebyone, num_classes, 2, 2, name="trans1")
    trans1 = tf.layers.conv2d_transpose(onebyone, 512, 2, 2, name="trans1")
    #skip1 = tf.add(trans1, helper.conv_1x1(vgg_layer4_out, num_classes))
    skip1 = tf.add(trans1, vgg_layer4_out)
    #skip1 =  trans1
    trans2= tf.layers.conv2d_transpose(skip1, 256, 2, 2, name="trans2")
    #trans2= tf.layers.conv2d_transpose(skip1, num_classes, 2, 2, name="trans2")
    #skip2 = tf.add(trans2, helper.conv_1x1(vgg_layer3_out, num_classes))
    skip2 = tf.add(trans2, vgg_layer3_out)
    #skip2 =  trans2
    #nn_last_out = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), name="trans3")
    #nn_last_out = tf.layers.conv2d_transpose(skip2, 256, 8, 8, name="trans3")
    nn_last_out = tf.layers.conv2d_transpose(skip2, num_classes, 8, 8, name="trans3")

    #debug shapes  return onebyone, trans1, trans2, nn_last_out
    return nn_last_out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    batch_gen = get_batches_fn(batch_size)
    for step in range(epochs):
       # print("setp ", step)
       batch = batch_gen
       try:
           batch = next(batch_gen)
       except TypeError:
           print(' batch_gen not iteratable')
       except StopIteration:
           print(' batch done')
           batch_gen = get_batches_fn(batch_size)
       else:
           _, loss = sess.run([train_op, cross_entropy_loss],
                              feed_dict={input_image: batch[0],
                                         correct_label: batch[1],
                                         keep_prob: 0.5,
                                         # learning_rate: 0.0001 -> loss = 1})
                                         # learning_rate: 0.00001 -> loss = 0.69})
                                         # learning_rate: 0.01 -> loss = 0.46})
                                         # learning_rate: 0.001 -> 0.7})
                                         learning_rate: 0.0001})
           if step % 5 == 0:
               print ("epochs: ", step, " loss: ", loss)
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    #image_shape = (260, 576)
    image_shape = (256, 512)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    correct_label = tf.placeholder(tf.float32, shape = (None, image_shape[0], image_shape[1], 2))
    learning_rate = tf.placeholder(tf.float32)

    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        #debug shapes onebyone, trans1, trans2, nn_last_out = layers(layer3, layer4, layer7, num_classes)
        nn_last_out = layers(layer3, layer4, layer7, num_classes)

        ### TODO
        logits, train_op, cross_entropy_loss = \
            optimize(nn_last_out, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        #batch_size = 1;
        batch_size = 20;
        epochs = 1000;
        init = tf.global_variables_initializer()
        sess.run(init)

        """ ### debug the dynamic shapes
        sess_out = sess.run([tf.shape(layer3),
                         tf.shape(layer4),
                         tf.shape(layer7),
                         tf.shape(onebyone),
                         tf.shape(trans1),
                         tf.shape(trans2),
                         tf.shape(nn_last_out),
                         tf.shape(logits),
                         ], feed_dict={image_input: np.zeros(
                                          (1, image_shape[0], image_shape[1], 3)),
                                       'keep_prob:0': 0.5})
        print ("*** dyn shapes ")
        names = ["layer3", "layer4", "layer7", "onebyone",
                 "trans1", "trans2", "nn_last_out", "logtis" ]
        for n, s in zip(names, sess_out):
            print(n, s)
        """

        train_nn(sess, epochs, batch_size, get_batches_fn,
                 train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
