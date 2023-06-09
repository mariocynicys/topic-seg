"""Preprocess the wiki data by removing the stop words, punctuation and
irrelevant parts to make it look more like an ASR output."""

import os
import nltk
import utils
import config
import string
from typing import Tuple, List

useless_tags = {"***LIST***"}
try:
    stop_words = set(nltk.corpus.stopwords.words('english')).union(useless_tags)
except:
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english')).union(useless_tags)

def recoverable_clean_section(section: str, vocab: dict) -> Tuple[str, List[str]]:
    """This is similar to clean_section but it stores the information about the original
    structure of the text stream so we can recover the correct splits in the input.
    """
    # Remove any punctuation.
    for p in string.punctuation:
        section = section.replace(p, ' ')
    words = [w.lower().strip() for w in section.split()]
    words = [w for w in words if w]
    original, clean_sec = [], []
    for word in words:
        ascii_word = utils.make_ascii(word)
        # Remove stop words, unknown words and non-ascii words.
        if not ascii_word or ascii_word in stop_words or ascii_word not in vocab:
            # 0 indicates that we missed this word.
            original.append(word)
        else:
            # 1 indicates that we took this word into account.
            original.append(word)
            clean_sec.append(ascii_word)
    return ' '.join(clean_sec), original

def clean_section(section: str, vocab: dict) -> str:
    return recoverable_clean_section(section, vocab)[0]

def process_doc(document: str, vocab: dict):
    lines = [l for l in document.split('\n') if l]

    # Divide the document into sections.
    sections, i = [], 0
    while i < len(lines):
        if lines[i].startswith(config.section_start):
            i += 1
            start = i
            while i < len(lines) and not lines[i].startswith(config.section_start):
                i += 1
            end = i
            sections.append(' '.join(lines[start:end]))
        else:
            i += 1

    # Rejoin the sections together.
    document = []
    for section in sections:
        section = clean_section(section, vocab)
        # Omit sections that are very short (doesn't meet the min words per sec requirement).
        if len(section.split()) >= config.min_words_per_section:
            document.append(section)
    return "\n\n".join(document)

def preprocess_wiki(wiki_path):
    # Only grab files with no extension (no .preprocessed/.tf/.anything files).
    files = utils.get_files_rec(wiki_path, forbidden=['.'])
    vocab = utils.load_wordvecs(config.word2vec_file_path)
    print("\nPreprocessing the files in", os.path.abspath(wiki_path))
    for file in utils.bar(files):
        document = open(file).read()
        open(file + ".preprocessed", 'w').write(process_doc(document, vocab))


if __name__ == "__main__":
    import sys
    preprocess_wiki(sys.argv[1])
