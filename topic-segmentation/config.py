"""A file containing the global configuration parameters for the model."""


def is_boundary(segmentations: list) -> bool:
    """A function the tells whether this segment should be a topic boundary or not.
    `segmentations` is a list of segmentation probabilities for the same segment but across multiple snippets.
    """
    import numpy as np
    return np.mean(np.sort(segmentations)[-5:]) >= 0.35

word2vec_file_path: str = "data/embeddings/wiki1m-trimmed.vec"
"""Where the word2vec embeddings at. This should be the path to a .vec embeddings file."""

min_words_per_section: int = 100
"""The minimum number of words a section in the training set has to be to be considered."""

section_start: str = "======="
"""The marker indicating the start of a new section in the training set."""

segment_length: int = 20
"""The size of the augmented sentence. Treat each `segment_size` words as one sentence."""

snippet_length: int = 14
"""How many sentences/segments form a snippet? (this is the K parameter in the 2LT paper)"""

snippet_stride: int = snippet_length // 2
"""The stride by which we stride to create snippets. Note that this stride is always set to 1 when predicting."""

positional_embeddings_length = 20
"""The size of the positional embeddings. This is used for segment level and snippet level positions."""

epochs: int = 5
"""The number of training epochs."""

model_store: str = "model.weights"
"""A directory where the trained model is stored and pulled from."""

tfrecord_tmpl: str = "data_$.tfrecord"
"""A file path template where tf record files (dataset) are stored in."""

segments_per_tfrecord: int = 20_000
"""The maximum number of segments to write to each tf record file."""

batch_size: int = 32
"""The training sample batch size."""

learning_rate: float = 0.0001
"""The model's learning rate."""

token_transformer_params: dict = {
    # Self attention params.
    "num_layers": 3,
    "num_attention_heads": 4,
    "attention_dropout": 0.01,
    # FFN params.
    "inner_dim": 1024,
    "inner_dropout": 0.01,
    # Post FFN dropout.
    "output_dropout": 0.01,
}
"""Token transformer parameters."""

sentence_transformer_params: dict = {
    "num_layers": 3,
    "num_attention_heads": 4,
    "attention_dropout": 0.01,
    "inner_dim": 1024,
    "inner_dropout": 0.01,
    "output_dropout": 0.01,
}
"""Sentence/Segment transformer parameters."""


def _check_config():
    assert word2vec_file_path.endswith(".vec"), "A word2vec file must end with a .vec extension."
    assert '$' in tfrecord_tmpl, "A tf-record template must contain $ character."
    assert positional_embeddings_length % 2 == 0, "Use an even number for the positional embeddings length since half the size is used for segment level positions and the other half for snippet level positions."
    assert min_words_per_section >= segment_length, "A training section can't be less than one segment (augmented sentence) length."
    keys = ["num_layers", "num_attention_heads", "inner_dim"]
    dicts = [token_transformer_params, sentence_transformer_params]
    for dictionary in dicts:
        for key in keys:
            if key not in dictionary:
                raise RuntimeError("Parameter '{}' is missing in one or more of the parameter dictionaries.".format(key))

# Don't comment this out.
# A broken config will break some assumptions that are taken in the code.
# Instead, accommodate your configuration be comply with these assumptions.
_check_config()


# Some cuda/tensorflow environment variables to set.
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'