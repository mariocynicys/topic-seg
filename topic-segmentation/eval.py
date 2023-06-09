import os
import sys
import utils
import config
import recordgen
import preprocess
from TLT import *


config.epochs = 1

def eval(tfrecord_tmpl):
    print("Evaluating...")
    weights = utils.load_model_weights(config.model_store)
    model = TwoLevelTransformerModel(weights)
    return model.evaluate(recordgen.get_dataset(tfrecord_tmpl))


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
