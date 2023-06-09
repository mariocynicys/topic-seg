import os
import utils
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

NEW_TOPIC_MARKER = "NEW_TOPIC_MARKER"


def read_files(files):
    for file in files:
        yield open(file).read().replace("\n\n", " " + NEW_TOPIC_MARKER + " ")

def segment_generator(reader):
    segment = []
    for doc in reader:
        # A new document is considered a topical change.
        had_marker = True
        for word in doc.split():
            if word == NEW_TOPIC_MARKER:
                had_marker = True
            else:
                segment.append(word)
                if len(segment) == config.segment_length:
                    yield (segment, had_marker)
                    # Reset the segment and the marker.
                    segment, had_marker = [], False

def write_segment(segment, writer, vocab):
    words, topic_boundary = segment
    # All the words should be in our vocabulary since we eliminated unknown words while preprocessing.
    tokens = [vocab[w] for w in words]
    record = tf.train.SequenceExample()
    record.context.feature["topic_bound"].int64_list.value.append(topic_boundary)
    tok_seq = record.feature_lists.feature_list["token_sequence"]
    for token in tokens:
        tok_seq.feature.add().int64_list.value.append(token)
    writer.write(record.SerializeToString())

def record_gen_predict(asr: str, record_path_template: str):
    """Same as record_gen but works on only one input document.
    More suitable for predicting.
    """
    vocab = utils.load_wordvecs(config.word2vec_file_path)
    writer = tf.io.TFRecordWriter(record_path_template.replace('$', '0'))
    for segment in segment_generator([asr]):
        write_segment(segment, writer, vocab)
    print("Generated records are stored in", os.path.abspath(record_path_template))

def record_gen(input_dir_path: str, record_path_template: str):
    utils.delete_template(record_path_template)
    files = utils.get_files_rec(input_dir_path, must_contain=[".preprocessed"])
    if len(files) == 0:
        print("WARNING: There are no .preprocessed files in", input_dir_path)
    print("\nGenerating TF records for the preprocessed files in", os.path.abspath(input_dir_path))
    # Don't load all the files in memory at once, create a lazy file reader instead.
    reader = read_files(utils.bar(files))
    vocab = utils.load_wordvecs(config.word2vec_file_path)
    for index, segment in enumerate(segment_generator(reader)):
        # Refresh the writer to parallelize the IO as much as possible when reading.
        if index % config.segments_per_tfrecord == 0:
            writer = tf.io.TFRecordWriter(record_path_template.replace('$', str(index // config.segments_per_tfrecord)))
        write_segment(segment, writer, vocab)
    print("Generated records are stored in", os.path.abspath(record_path_template))

def input_fn(record_path_template: str):
    def make_dataset(deserializer):
        paths, index = [], 0
        while os.path.exists(record_path_template.replace('$', str(index))):
            paths.append(record_path_template.replace('$', str(index)))
            index += 1
        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.repeat(config.epochs)
        dataset = dataset.map(deserializer, num_parallel_calls=128)
        # Create a sliding window over the data with a `snippet_stride` slider size.
        dataset = dataset.window(size=config.snippet_length, shift=config.snippet_stride, drop_remainder=True)
        # Join the windows back together. (This outputs every window batched instead of being a `_VariantDataset`).
        dataset = dataset.flat_map(lambda window: window.batch(config.snippet_length))
        # Batch every `config.batch_size` windows together as one training example.
        dataset = dataset.batch(config.batch_size, drop_remainder=True)
        # Buffer some batches to lower the IO latency.
        dataset = dataset.prefetch(20)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    def deserialize_labels(sr):
        return tf.io.parse_single_sequence_example(serialized=sr, context_features={"topic_bound": tf.io.FixedLenFeature([], dtype=tf.int64)})[0]["topic_bound"]
    def deserialize_features(sr):
        return tf.io.parse_single_sequence_example(serialized=sr, sequence_features={"token_sequence": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)})[1]["token_sequence"]
    def deserialize(serialized_record):
        context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
            serialized=serialized_record,
            context_features={"topic_bound": tf.io.FixedLenFeature([], dtype=tf.int64)},
            sequence_features={"token_sequence": tf.io.FixedLenSequenceFeature([], dtype=tf.int64)},
        )
        return (sequence_parsed["token_sequence"], context_parsed["topic_bound"])

    return make_dataset(deserialize_features), make_dataset(deserialize_labels)

def get_batch_count(record_path_template: str):
    paths, index = [], 0
    while os.path.exists(record_path_template.replace('$', str(index))):
        paths.append(record_path_template.replace('$', str(index)))
        index += 1
    # Assuming every tf record is full. This is a pessimistic estimation.
    # Also assuming `config.segments_per_tfrecord` have changed sense the record generation.
    segment_count = index * config.epochs * config.segments_per_tfrecord
    snippet_count = segment_count // config.snippet_stride
    batch_count = snippet_count // config.batch_size
    return batch_count


if __name__ == "__main__":
    import sys
    input_dir_path = sys.argv[1]
    record_gen(input_dir_path, os.path.join(input_dir_path, config.tfrecord_tmpl))
