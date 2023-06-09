#!/bin/bash

echo "Downloads and prepares the needed data. Run only once."

# Refuse to run on a dirty data direcotry.
mkdir data || exit
cd data

# Make sure gdown is installed
pip install gdown

# Download wiki_727K.tar.bz2
gdown 124oH8sx9i3hjSNpl7UJUP4moQeyZLvQj
# Decompress the tar file.
tar -xjf wiki_727K.tar.bz2
# tar omits the `K` for some reason (at least on my machine), add it back.
mv wiki_727 wiki_727K

mkdir embeddings
cd embeddings

# Download crawl-300 embeddings
gdown 1-5RIV4UCYEShfrS3oBwLgNYEjQiVtNaX
# Download wiki-300 ebmeddings
gdown 1-5mclrHsSHC-xA-luF3Mz20xQvC-LivC
