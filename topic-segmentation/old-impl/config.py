"""A file containing the global configuration parameters for the model."""


segmentation_threshold: float = 0.3
"""The threshold based one we decide a segment/augmented sentence is a topic boundary or not."""

word2vec_file_path: str = "data/embeddings/wiki1m-trimmed.vec"
"""Where the word2vec embeddings at. This should be the path to a .vec embeddings file."""

min_words_per_section: int = 100
"""The minimum number of words a section in the training set has to be to be considered."""

section_start: str = "======="
"""The marker indicating the start of a new section in the training set."""

random_seed: int = 849
"""A random seed, can be any number. Controls some randomness in the model."""

segment_length: int = 20
"""The size of the augmented sentence. Treat each `segment_size` words as one sentence."""

snippet_length: int = 14
"""How many sentences/segments form a snippet? (this is the K parameter in the 2LT paper)"""

snippet_stride: int = snippet_length // 2
"""The stride by which we stride to create snippets. Note that this stride is always set to 1 when predicting."""

positional_embeddings_length = 20
"""The size of the positional embeddings. This is used for segment level and snippet level positions."""

epochs: int = 10000
"""The number of training epochs."""

model_dir: str = "data/model/"
"""A directory where the trained model is stored and pulled from."""

tfrecord_tmpl: str = "data_$.tfrecord"
"""A file path template where tf record files (dataset) are stored in."""

segments_per_tfrecord: int = 20_000
"""The maximum number of segments to write to each tf record file."""

batch_size: int = 8
"""The training sample batch size."""

learning_rate: float = 0.000001
"""The model's learning rate."""

token_transformer_training_params: dict = {
    "num_hidden_layers": 6,
    "num_heads": 4,
    "filter_size": 1024,
    "relu_dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "allow_ffn_pad": True
}
"""Token transformer training parameters."""

sentence_transformer_training_params: dict = {
    "num_hidden_layers": 6,
    "num_heads": 4,
    "filter_size": 1024,
    "relu_dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "allow_ffn_pad": True
}
"""Sentence/Segment transformer training parameters."""

token_transformer_prediction_params: dict = {
    "num_hidden_layers": 6,
    "num_heads": 4,
    "filter_size": 1024,
    "relu_dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "allow_ffn_pad": True
}
"""Token transformer prediction parameters."""

sentence_transformer_prediction_params: dict = {
    "num_hidden_layers": 6,
    "num_heads": 4,
    "filter_size": 1024,
    "relu_dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_postprocess_dropout": 0.1,
    "allow_ffn_pad": True
}
"""Sentence/Segment transformer prediction parameters."""


def _check_config():
    assert word2vec_file_path.endswith(".vec"), "A word2vec file must end with a .vec extension."
    assert '$' in tfrecord_tmpl, "A tf-record template must contain $ character."
    assert positional_embeddings_length % 2 == 0, "Use an even number for the positional embeddings length since half the size is used for segment level positions and the other half for snippet level positions."
    assert min_words_per_section >= segment_length, "A training section can't be less than one segment (augmented sentence) length."
    keys = ["num_hidden_layers", "num_heads", "filter_size", "relu_dropout", "attention_dropout", "layer_postprocess_dropout", "allow_ffn_pad"]
    dicts = [token_transformer_training_params, sentence_transformer_training_params, token_transformer_prediction_params, sentence_transformer_prediction_params]
    for dictionary in dicts:
        for key in keys:
            if key not in dictionary:
                raise RuntimeError("Parameter '{}' is missing in one or more of the parameter dictionaries.".format(key))

# Don't comment this out.
# A broken config will break some assumptions that are taken in the code.
# Instead, accommodate your configuration be comply with these assumptions.
_check_config()
