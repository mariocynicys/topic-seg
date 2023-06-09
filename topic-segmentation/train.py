import os
import sys
import uuid
import utils
import config
import recordgen
import preprocess
from tensorflow import keras
from TLT import *

model = None

class StoreWeightsCallback(keras.callbacks.Callback):
    run_id = str(uuid.uuid4()).split('-')[0]

    def on_epoch_end(self, epoch, _=None):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        file_name = os.path.join("checkpoints", self.run_id + '-e' + str(epoch) + ".weights")
        utils.store_model_weights(model, file_name)


def train(tfrecord_tmpl):
    global model
    # Load the model from the file system if it is stored.
    if os.path.exists(config.model_store):
        print("Loading a model from the file system.")
        weights = utils.load_model_weights()
        model = TwoLevelTransformerModel(weights)
    else:
        print("Compiling a new model.")
        _, word_repr = utils.load_wordvecs(config.word2vec_file_path, load_embs=True)
        model = TwoLevelTransformerModel(word_repr=word_repr)

    batch_count = recordgen.get_batch_count(tfrecord_tmpl)
    print("Number of training batches =", batch_count)
    print("Number of training epochs =", config.epochs)
    print("Training...")

    dataset = recordgen.get_dataset(tfrecord_tmpl)
    try:
        model.fit(
            dataset, steps_per_epoch=batch_count,
            epochs=config.epochs, callbacks=[StoreWeightsCallback()]
        )
    except KeyboardInterrupt:
        if input("\nDo you wan't to save that model? (Y/n) ").lower().strip() != "n":
            utils.store_model_weights(model)
            print("Model weights saved to", config.model_store)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    tfrecord_tmpl = os.path.join(input_dir, config.tfrecord_tmpl)
    skip_preprocessing = (len(sys.argv) > 2)
    if not skip_preprocessing:
        preprocess.preprocess_wiki(input_dir)
        recordgen.record_gen(input_dir, tfrecord_tmpl)
    else:
        print("Skipping preprocessing!")
    train(tfrecord_tmpl)
