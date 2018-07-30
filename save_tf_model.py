import argparse
import os

import tensorflow as tf
import keras.backend as K

from models import get_model_f, get_model, load_best_weights_max
from constants import MODELS_PATH


def save(args):
    K.set_learning_phase(0)
    model_f = get_model_f(args.model_name)
    model = get_model(model_f)
    fold_wdir = load_best_weights_max(model, args.model_name, wdir=args.wdir)
    # meta_graph_def = tf.train.export_meta_graph(os.path.join(MODELS_PATH, 'tf', args.model_name + '.meta'))


    output = model.get_layer('output').output
    # in_rgb = model.get_layer('rgb').output
    # in_audio = model.get_layer('audio').output
    # in_all = tf.concat((in_rgb, in_audio), axis=1)

    tf.add_to_collection("predictions", output)
    print(output)
    print('Check', tf.get_collection('predictions'))

    in_all = model.get_layer('input_batch_raw').input
    tf.add_to_collection("input_batch_raw", in_all)
    print(in_all)
    print('Check', tf.get_collection('input_batch_raw'))

    num_frames = K.ones_like(output[:, 0])
    tf.add_to_collection("num_frames", num_frames)
    print(num_frames)
    print('Check', tf.get_collection('num_frames'))

    saver = tf.train.Saver(tf.global_variables())

    with K.get_session() as sess:
        sess.run(tf.global_variables_initializer())
        save_path = saver.save(sess, os.path.join(MODELS_PATH, 'tf', args.model_name, 'inference_model'))
        meta_graph_def = saver.export_meta_graph(clear_devices=True)


    	# tf.add_to_collection("global_step", 0)
        # tf.add_to_collection("loss", 0)
        # tf.add_to_collection("input_batch", 0)
        # tf.add_to_collection("video_id_batch", 0)
        # tf.add_to_collection("labels", 0)
        # tf.add_to_collection("summary_op", 0)


def load(args):
    os.path.join(MODELS_PATH, 'tf', args.model_name, "inference_model")
    meta_graph_location = checkpoint_file + ".meta"

    with tf.device("/cpu:0"):
      saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + checkpoint_file)
    saver.restore(sess, checkpoint_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='dense4')
    parser.add_argument('-w', '--wdir', type=str, default=None, help='weights dir, if None - load by model_name')
    args = parser.parse_args()

    args.model_name = 'dense4'
    args.use_tf_records = True
    args.out_file = 'dense4'

    save(args)
    # load(args)

