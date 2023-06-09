import os
import sys
import tqdm
import random
import subprocess
import config
import numpy as np
from typing import List, Union, Tuple, Any


def get_files_rec(path: str, forbidden=[], must_contain=[]) -> List[str]:
    def _inner(path: str):
        files = []
        parent = path
        for path in os.listdir(parent):
            path = os.path.join(parent, path)
            if os.path.isdir(path):
                files.extend(_inner(path))
            else:
                files.append(path)
        return files
    files = _inner(path)
    files = [file for file in files if all(forbid not in file for forbid in forbidden)]
    if len(must_contain):
        files = [file for file in files if any(must in file for must in must_contain)]
    files.sort()
    random.seed(config.random_seed)
    random.shuffle(files)
    return files

def delete_template(tmpl: str):
    i = 0
    while os.path.exists(tmpl.replace('$', str(i))):
        os.remove(tmpl.replace('$', str(i)))
        i += 1

def bar(it, *args, **kwargs) -> tqdm.tqdm:
    TQDM_DEFAULT_SETTINGS = {
        'colour': 'GREEN',
        'file': sys.stdout,
    }
    kwargs = {**TQDM_DEFAULT_SETTINGS, **kwargs}
    try:
        if kwargs.get("total") is None:
            kwargs["total"] = len(it)
    except:
        pass
    return tqdm.tqdm(it, *args, **kwargs)

def load_wordvecs(path: str, load_embs=False) -> Union[dict, Tuple[dict, Any]]:
    """A .vec word embeddings file to load the vocabulary and their embeddings from."""
    npz_path = path.removesuffix(".vec") + ".npz"
    if not os.path.exists(npz_path):
        print("\nWord embeddings are not prepared! Preparing them now.")
        print("This will take some time...")
        vec2npz(path)
    word2vec = np.load(npz_path)
    words = dict((w, i) for i, w in enumerate(word2vec["words"]))
    if not load_embs:
        return words
    # Loading the embeddings takes a lot of time.
    return words, word2vec["embeddings"]

def vec2npz(path: str, extra_vocab=[]):
    """Converts a .vec embeddings to vocabulary and embeddings lists and stores
    them as a compressed numpy array on the same path with .npz extension.
    """
    with open(path) as f:
        w_count, dim = f.readline().split()
        w_count, dim = int(w_count), int(dim)
        total_count = (w_count + len(extra_vocab))
        words = [""] * total_count
        embeddings = np.zeros((total_count, dim), dtype=np.float32)
        for i, line in enumerate(bar(f, total=w_count)):
            word, emb = line.split(maxsplit=1)
            words[i] = word
            embeddings[i] = np.array(emb.split(), dtype=np.float32)
        # Actual count can be less than the specified count in case of the vector file being trimmed.
        actual_count = i + 1

    np.random.seed(config.random_seed)
    for i, word in enumerate(extra_vocab):
        words[actual_count + i] = word
        # This generates a random 300d array of values in range [-0.99, 0.99]
        embeddings[actual_count + i] = 1.98 * np.random.rand(300).astype(np.float32) - 0.99
    actual_count += len(extra_vocab)

    # Convert the words array to a numpy array before storing.
    words = np.array(words)

    # Squeeze to the actual count.
    embeddings = embeddings[:actual_count]
    words = words[:actual_count]
    print("Storing the prepared word embeddings in a .npz file.")
    npz_path = path.removesuffix(".vec") + ".npz"
    np.savez(npz_path, embeddings=embeddings, words=words)
    print("Prepared word embeddings are now stored in: ", os.path.abspath(npz_path))

def trim_vocab_from(path: str, word2vec_file_path: str) -> str:
    """Trims `word2vec_file_path` to a set that has only the words found in files in `path`.
    The word2vec model we use have about one million unique words. But in spoken and written language,
    people use less than 1% of that (10k words).
    This helps in memory usage.
    """
    vocab, embeddings = load_wordvecs(word2vec_file_path, load_embs=True)
    vocab_set = set(vocab.keys())
    words_in_files = set()
    files = get_files_rec(path, must_contain=[".preprocessed"])
    for file in files:
        words = open(file).read().split()
        words_in_files.update(words)
    # Keep only the vocabs which appeared in the files we have.
    vocab_set = vocab_set.intersection(words_in_files)
    vocab = dict((word, vocab[word]) for word in vocab_set)
    new_vocab = [""] * len(vocab)
    new_embeddings = np.zeros((len(new_vocab), embeddings.shape[1]), dtype=np.float32)
    # Create the new vocab and embeddings maps.
    for i, word in enumerate(vocab):
        new_vocab[i] = word
        new_embeddings[i] = embeddings[vocab[word]]
    # Store the trimmed vocab.
    npz_path = word2vec_file_path.removesuffix(".vec") + "-trimmed.npz"
    np.savez(npz_path, embeddings=new_embeddings, words=new_vocab)
    print(f"Trimmed word embeddings (size={len(new_vocab)}) are now stored in: ", os.path.abspath(npz_path))
    return npz_path.removesuffix(".npz") + ".vec"

def make_ascii(word: str) -> str:
    return bytes(word, encoding='utf-8').decode('ascii', 'ignore').strip()

def get_sub(link: str, fname: str):
    """Downloads the auto-generated subtitles from the youtube link `link` and persists them in `file`."""
    subprocess.run(f"yt-dlp --write-auto-sub --skip-download \"{link}\" -o {fname}", shell=True).check_returncode()
    subprocess.run(f"ffmpeg -y -i {fname}.en.vtt {fname}.srt", shell=True).check_returncode()

    with open(f"{fname}.srt") as file:
        lines = file.readlines()
        good_lines = []
        i = 0
        while i != len(lines):
            try:
                int(lines[i])
                if lines[i + 1].__contains__('-->'):
                    i += 2
            except:
                pass
            good_lines.append(lines[i])
            i += 1
        with open(f"{fname}.txt", 'w') as file2:
            for line in good_lines:
                file2.write(line)

    # SRT files contain duplicate lines for some reason, delete them.
    with open(f"{fname}.txt") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != '']
        last_line = None
        non_dup_lines = []
        for line in lines:
            if last_line == line:
                continue
            non_dup_lines.append(line)
            last_line = line

    with open(f"{fname}.asr", "w") as file:
        file.write(' '.join(non_dup_lines))

    # Remove the intermediate files.
    os.remove(f"{fname}.txt")
    os.remove(f"{fname}.srt")
    os.remove(f"{fname}.en.vtt")
