import os
import sys
import utils
import shutil
import config
import recordgen
import preprocess
import numpy as np
import tensorflow as tf
from transformer.model import transformer, model_utils


def model_fn(features, labels, mode, params):
    prints = []
    def out(*args, **kwargs):
        prints.append(tf.print(args, kwargs, output_stream=sys.stdout))
    def shape(t, n):
        print(n + ":", t)
    def uniform(s):
        return tf.initializers.GlorotUniform()(s)

    predicting = mode == tf.estimator.ModeKeys.PREDICT
    if not predicting:
        TT_PARAMS = config.token_transformer_training_params
        ST_PARAMS = config.sentence_transformer_training_params
    else:
        TT_PARAMS = config.token_transformer_prediction_params
        ST_PARAMS = config.sentence_transformer_prediction_params

    embeddings = params["embeddings"]
    snippet_batch = tf.reshape(features, [config.batch_size, config.snippet_length, config.segment_length])
    shape(snippet_batch, "snippet_batch from features")

    # The distributions used for positional embeddings. Each, segment & snippet level positions take half the `config.positional_embeddings_length`.
    token_positional_embeddings_dist = tf.Variable(uniform([config.segment_length, config.positional_embeddings_length // 2]))
    sentence_positional_embeddings_dist = tf.Variable(uniform([config.snippet_length, config.positional_embeddings_length // 2]))
    shape(token_positional_embeddings_dist, "token_positional_embeddings_dist (variable)")
    shape(sentence_positional_embeddings_dist, "sentence_positional_embeddings_dist (variable)")

    # Arrays with the same shape as `snippet_batch` that has the nominal positions of the token on a segment level and snippet level.
    token_nominal_positions = np.tile(np.arange(config.segment_length), [config.batch_size, config.snippet_length, 1])
    sentence_nominal_positions = np.tile(np.tile(np.arange(config.snippet_length), [config.segment_length, 1]).T, [config.batch_size, 1, 1])
    shape(token_nominal_positions.shape, "token_nominal_positions (tile)")
    shape(sentence_nominal_positions.shape, "sentence_nominal_positions (tile)")

    # Look up the word and positional embeddings and concatenate them together as the final embeddings.
    snippet_batch_embeddings = tf.nn.embedding_lookup(embeddings, snippet_batch)
    shape(snippet_batch_embeddings, "snippet_batch_embeddings (without pos embs)")
    token_positional_embeddings = tf.nn.embedding_lookup(token_positional_embeddings_dist, token_nominal_positions)
    sentence_positional_embeddings = tf.nn.embedding_lookup(sentence_positional_embeddings_dist, sentence_nominal_positions)
    shape(token_positional_embeddings, "token_positional_embeddings (after lookup)")
    shape(sentence_positional_embeddings, "sentence_positional_embeddings (after lookup)")
    snippet_batch_embeddings = tf.concat([snippet_batch_embeddings, token_positional_embeddings, sentence_positional_embeddings], axis=3)
    shape(snippet_batch_embeddings, "snippet_batch_embeddings (after concat with pos embs)")

    # Token-level transformer (TT):
    hidden_size = embeddings.shape[1] + config.positional_embeddings_length
    tt_input = tf.reshape(snippet_batch_embeddings, [config.batch_size * config.snippet_length, config.segment_length, hidden_size])
    shape(tt_input, "tt_input")
    TT_PARAMS.update({"hidden_size": hidden_size})
    tt_trans = transformer.EncoderStack(TT_PARAMS, mode)
    # Since we don't have any padding in our input, the padding is gonna be all zeros.
    attention_padding = tf.zeros([config.batch_size * config.snippet_length, config.segment_length])
    attention_bias = model_utils.get_padding_bias(tf.zeros([config.batch_size * config.snippet_length, config.segment_length]), padding_value=-1)
    tt_output = tt_trans(tt_input, attention_bias, attention_padding)
    shape(tt_output, "tt_output")
    # We will use the first and last token to represent the sentence.
    sentence_embeddings = tt_output[:, 0, :] #tf.concat([tt_output[:, 0, :], tt_output[:, -1, :]], axis=1)
    shape(sentence_embeddings, "sentence_embeddings (selected stuff from tt_output")

    # Sentence-level transformer (ST):
    hidden_size = sentence_embeddings.shape[1]
    st_input = tf.reshape(sentence_embeddings, [config.batch_size, config.snippet_length, hidden_size])
    shape(st_input, "st_input")
    ST_PARAMS.update({"hidden_size": hidden_size})
    st_trans = transformer.EncoderStack(ST_PARAMS, mode)
    # No padding here as well.
    attention_padding = tf.zeros([config.batch_size, config.snippet_length])
    attention_bias = model_utils.get_padding_bias(tf.zeros([config.batch_size, config.snippet_length]), padding_value=-1)
    st_output = st_trans(st_input, attention_bias, attention_padding)
    shape(st_output, "st_output")

    # Segmentation classifier:
    seg_classifier_w = tf.Variable(uniform([st_output.shape[2], 2]))
    seg_classifier_b = tf.Variable(uniform([2]))
    shape(seg_classifier_w, "seg_classifier_w")
    shape(seg_classifier_b, "seg_classifier_b")
    seg_probabilities = tf.nn.softmax(tf.add(tf.tensordot(st_output, seg_classifier_w, axes=[[2], [0]]), seg_classifier_b))
    shape(seg_probabilities, "seg_probabilities")

    if not predicting:
        # Prepare segment labels:
        labels = tf.reshape(labels, [config.batch_size, config.snippet_length])
        label_2d_fn = lambda x: tf.cond(tf.equal(x, 1), lambda: tf.constant([0., 10.]), lambda: tf.constant([1., 0.]))
        segment_labels = tf.map_fn(lambda x: tf.map_fn(label_2d_fn, x, dtype=tf.float32), labels, dtype=tf.float32)
        shape(segment_labels, "segment_labels")

        # Define the loss:
        segmentation_loss = -1 * tf.reduce_sum(tf.multiply(tf.math.log(seg_probabilities), segment_labels))
        shape(segmentation_loss, "segmentation_loss")
        tf.summary.scalar("Segmentation Loss", segmentation_loss)

        optimizer_seg = tf.compat.v1.train.AdamOptimizer(learning_rate=config.learning_rate)
        # Make this statement depend on the prints we have above to not optimize them away.
        with tf.control_dependencies(prints):
            train_op = optimizer_seg.minimize(segmentation_loss, tf.compat.v1.train.get_global_step())

        print("Model defined.")
        return tf.estimator.EstimatorSpec(mode, loss=segmentation_loss, train_op=train_op)
    else:
        return tf.estimator.EstimatorSpec(mode, predictions=seg_probabilities)


def train(tfrecord_tmpl):
    print("Loading the word2vec model...")
    _, embeddings = utils.load_wordvecs(config.word2vec_file_path, load_embs=True)
    model_conf = tf.estimator.RunConfig(model_dir=config.model_dir)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=model_conf, params={"embeddings": embeddings})
    print("Training...")
    estimator.train(input_fn=lambda: recordgen.input_fn(tfrecord_tmpl))


if __name__ == "__main__":
    input_dir = sys.argv[1]
    tfrecord_tmpl = os.path.join(input_dir, config.tfrecord_tmpl)
    skip_preprocessing = (len(sys.argv) > 2)
    if not skip_preprocessing:
        preprocess.preprocess_wiki(input_dir)
        recordgen.record_gen(input_dir, tfrecord_tmpl)
        shutil.rmtree(config.model_dir, ignore_errors=True)
    else:
        print("Skipping preprocessing!")
    train(tfrecord_tmpl)
