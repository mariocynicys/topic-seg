import os
import sys
import utils
import shutil
import config
import tempfile
import recordgen
import preprocess
import numpy as np
import tensorflow as tf
from train import model_fn


config.epochs = 1
config.batch_size = 1
config.snippet_stride = 1


def predict(input_dir):
    input_files = os.listdir(input_dir)
    assert len(input_files) == 1, "The input directory must contain only one ASR file to segment."
    input_file = os.path.join(input_dir, input_files[0])

    # Load the vocabulary and their embeddings.
    vocab, embeddings = utils.load_wordvecs(config.word2vec_file_path, load_embs=True)

    clean_asr, recovery_list = preprocess.recoverable_clean_section(open(input_file).read().strip(), vocab)

    tfrecord_tmpl = os.path.join(input_dir, config.tfrecord_tmpl)
    recordgen.record_gen_predict(clean_asr, tfrecord_tmpl)

    model_conf = tf.estimator.RunConfig(model_dir=config.model_dir, log_step_count_steps=1)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=model_conf, params={"embeddings": embeddings})
    snippet_estimations = estimator.predict(input_fn=lambda: recordgen.input_fn(tfrecord_tmpl))

    clean_asr = clean_asr.split()
    segmented_clean_asr = [clean_asr[x:x + config.segment_length] for x in range(0, len(clean_asr), config.segment_length)]
    # Remove the last segment if it wasn't a full `config.segment_length`.
    if len(segmented_clean_asr[-1]) != config.segment_length:
        segmented_clean_asr.pop()
    snipped_clean_asr = [segmented_clean_asr[x:x + config.snippet_length] for x in range(0, len(segmented_clean_asr), config.snippet_stride)]
    # Remove the snippets which aren't a full `config.snippet_length`. Note that they are many and not only one since snippet_stride == 1.
    for i, snippet in enumerate(snipped_clean_asr):
        if len(snippet) != config.snippet_length:
            snipped_clean_asr = snipped_clean_asr[:i]
            break

    # Calculate the boundaries.
    boundaries = [[] for _ in range(len(segmented_clean_asr))]
    for snippet_index, segment_estimations in enumerate(snippet_estimations):
        for segment_index, segment_estimation in enumerate(segment_estimations):
            real_segment_index = snippet_index + segment_index
            boundaries[real_segment_index].append(segment_estimation[1] - segment_estimation[0])
    boundaries = [np.mean(x) >= config.segmentation_threshold for x in boundaries]

    # Recover the original text and inject the boundary markers.
    words = []
    original_asr_index = 0
    for segment, is_boundary in zip(segmented_clean_asr, boundaries):
        # If this segment is a boundary, prepend two line breaks before writing it.
        if is_boundary:
            words.append("\n\n")

        # Write the segment and sync it with the original word list.
        for word in segment:
            while word != utils.make_ascii(recovery_list[original_asr_index]):
                words.append(recovery_list[original_asr_index])
                original_asr_index += 1
            words.append(recovery_list[original_asr_index])
            original_asr_index += 1

    # Write the rest of the words that didn't make a full segment.
    for word in recovery_list[original_asr_index:]:
        words.append(word)

    open(os.path.basename(input_file) + ".segmented", 'w').write(' '.join(words))


if __name__ == "__main__":
    input_file = sys.argv[1]
    tmp_dir = tempfile.mkdtemp()
    print("Working inside", tmp_dir)
    shutil.copyfile(input_file, os.path.join(tmp_dir, os.path.basename(input_file)))
    predict(tmp_dir)
    shutil.rmtree(tmp_dir)
