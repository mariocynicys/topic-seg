import os
import sys
import utils
import config
import recordgen
import preprocess
import tensorflow as tf
from train import model_fn


config.epochs = 1

def eval(tfrecord_tmpl):
    print("Loading the word2vec model...")
    _, embeddings = utils.load_wordvecs(config.word2vec_file_path, load_embs=True)
    model_conf = tf.estimator.RunConfig(model_dir=config.model_dir)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=model_conf, params={"embeddings": embeddings})
    print("Evaluating...")
    metrics = estimator.evaluate(input_fn=lambda: recordgen.input_fn(tfrecord_tmpl))
    return metrics


if __name__ == "__main__":
    input_dir = sys.argv[1]
    tfrecord_tmpl = os.path.join(input_dir, config.tfrecord_tmpl)
    skip_preprocessing = (len(sys.argv) > 2)
    if not skip_preprocessing:
        preprocess.preprocess_wiki(input_dir)
        recordgen.record_gen(input_dir, tfrecord_tmpl)
    else:
        print("Skipping preprocessing!")
    metrics = eval(tfrecord_tmpl)
    print(metrics)
