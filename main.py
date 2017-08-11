import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    #print("global vars: ", tf.global_variables())
    #print("local vars: ", tf.local_variables())
    #for op in default_graph.get_operations():
    #    print(op.name)
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
    #KFC onebyone = tf.layers.conv2d(vgg_layer7_out, num_classes, 1 , 1)
    onebyone = tf.layers.conv2d(vgg_layer7_out, 4096, 1 , 1)
    #KFC trans1 = tf.layers.conv2d_transpose(onebyone, 512, 4, strides=(2, 2), name="trans1")
    trans1 = tf.layers.conv2d_transpose(onebyone, 512, 2, strides=(2, 2), name="trans1")
    skip1 = tf.add(trans1, vgg_layer4_out)
    trans2= tf.layers.conv2d_transpose(skip1, 256, 2, strides=(2, 2), name="trans2")
    skip2 = tf.add(trans2, vgg_layer3_out)
    #KFC nn_last_out = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), name="trans3")
    nn_last_out = tf.layers.conv2d_transpose(skip2, num_classes, 8, strides=(8, 8), name="trans3")

    return onebyone, trans1, trans2, nn_last_out
#KFC tests.test_layers(layers)


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
#KFC tests.test_optimize(optimize)


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
    #print (train_op)
    batch_gen = get_batches_fn(batch_size)
    for step in range(epochs):
       print("setp ", step)
       batch = next(batch_gen)
       print("batch.shape: ", batch[0].shape, batch[1].shape)
       #global_vars = tf.global_variables()
       #for v in global_vars:
       #    print(v)
       #default_graph = tf.get_default_graph()
       #ii = default_graph.get_tensor_by_name("image_input:0")
       #sess.run(ii, feed_dict={})
       #print(ii)
       print (train_op.run(
       #print (sess.run([train_op, cross_entropy_loss],
           feed_dict={input_image: batch[0],
                      correct_label: batch[1],
                      keep_prob: 0.5,
                      learning_rate: 0.01
                      } ))
# KFC tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    #KFC image_shape = (260, 576)
    image_shape = (256, 256)
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
    config.gpu_options.per_process_gpu_memory_fraction =0.5

    correct_label = tf.placeholder(tf.float32, shape = (None, image_shape[0], image_shape[1], 2))
    learning_rate = tf.placeholder(tf.float32)

    with tf.Session(config=config) as sess:
        #print("global vars: ", tf.global_variables())
        #print("local vars: ", tf.local_variables())
        #tf.report_uninitialized_variables();
        #tf.report_uninitialized_variables();
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        onebyone, trans1, trans2, nn_last_out = layers(layer3, layer4, layer7, num_classes)
        #last_layer = tf.reshape(layer_out, (-1, num_classes))
        ### TODO

        logits, train_op, cross_entropy_loss = \
            optimize(nn_last_out, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        batch_size = 1;
        epochs = 1;
        ### train_data_dir = os.path.join(data_dir, 'data_road', 'training')
        init = tf.global_variables_initializer()
        sess.run(init)
        default_graph = tf.get_default_graph()

        global_vars = tf.global_variables()
        for v in global_vars:
            print(v)

        #print("local vars: ", tf.local_variables())

        batch_gen = get_batches_fn(batch_size) # debug only
        batch = next(batch_gen)  # debug only
        sess_out = sess.run([tf.shape(layer3),
                         tf.shape(layer4),
                         tf.shape(layer7),
                         tf.shape(onebyone),
                         tf.shape(trans1),
                         tf.shape(trans2),
                         tf.shape(nn_last_out),
                         tf.shape(logits),
                         ], feed_dict={image_input: batch[0],
                                       keep_prob: 0.5})

        print ("*** dyn shapes ")
        names = ["layer3", "layer4", "layer7", "onebyone",
                 "trans1", "trans2", "nn_last_out", "logtis" ]
        for n, s in zip(names, sess_out):
            print(n, s)

        #train_nn(sess, epochs, batch_size, helper.gen_batch_function(train_data_dir, image_shape),
        #         train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)
        train_nn(sess, epochs, batch_size, get_batches_fn,
                 train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
