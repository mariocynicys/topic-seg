import os
import sys
import utils
import config
import datetime
import recordgen
import preprocess
from tensorflow import keras
from TLT import *

model = None

class StoreWeightsCallback(keras.callbacks.Callback):
    now = datetime.datetime.now()
    run_id = f"{now.month}-{now.day}_{now.hour}-{now.minute}"

    def on_epoch_end(self, epoch, _=None):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        file_name = os.path.join("checkpoints", self.run_id + '_e' + str(epoch) + ".weights")
        utils.store_model_weights(model, file_name)


def train(train_tfrecord_tmpl, val_tfrecord_tmpl):
    global model
    # Load the model from the file system if it is stored.
    if os.path.exists(config.model_store):
        print("Loading a model from the file system.")
        weights = utils.load_model_weights(config.model_store)
        model = TwoLevelTransformerModel(weights)
    else:
        print("Compiling a new model.")
        _, word_repr = utils.load_wordvecs(config.word2vec_file_path, load_embs=True)
        model = TwoLevelTransformerModel(word_repr=word_repr)

    batch_count = recordgen.get_batch_count(train_tfrecord_tmpl)
    print("Number of training batches =", batch_count)
    print("Number of training epochs =", config.epochs)
    print("Training...")

    train_dataset = recordgen.get_dataset(train_tfrecord_tmpl)
    val_dataset = recordgen.get_dataset(val_tfrecord_tmpl)
    try:
        model.fit(
            train_dataset, steps_per_epoch=batch_count,
            epochs=config.epochs, callbacks=[StoreWeightsCallback()],
            validation_data=val_dataset
        )
    except KeyboardInterrupt:
        if input("\nDo you wan't to save that model? (Y/n) ").lower().strip() != "n":
            utils.store_model_weights(model, config.model_store)
            print("Model weights saved to", config.model_store)


if __name__ == "__main__":
    training_dir, validation_dir = sys.argv[1], sys.argv[2]
    train_tfrecord_tmpl = os.path.join(training_dir, config.tfrecord_tmpl)
    val_tfrecord_tmpl = os.path.join(validation_dir, config.tfrecord_tmpl)
    skip_preprocessing = (len(sys.argv) > 3)
    if not skip_preprocessing:
        preprocess.preprocess_wiki(training_dir)
        recordgen.record_gen(training_dir, train_tfrecord_tmpl)
        preprocess.preprocess_wiki(validation_dir)
        recordgen.record_gen(validation_dir, val_tfrecord_tmpl)
    else:
        print("Skipping preprocessing!")
    train(train_tfrecord_tmpl, val_tfrecord_tmpl)
