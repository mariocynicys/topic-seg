# Topic-based speech (or document) segmentation
This repo implements [Two-Level Transformer and Auxiliary Coherence Modeling for Improved Text Segmentation](https://arxiv.org/abs/2001.00891) paper which holds the state of the art results in segmenting documents based on topic.

It is a supervised deep learning technique that uses `wiki_727K` dataset introduced in [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337)

# Approach
The assumption here is that document and speech aren't very different from each other in the topic segmentation domain. Speech generally is less coherent, less structured and uses a lesser set of words, but the approach taken by the paper shouldn't make it any weaker for speech segmentation.

Since we know ASR has a lot of failures and speech is generally less coherent, we removed the auxiliary coherence branch of the pipeline. Having it in place would likely trigger a lot of false positives because the training data (from wikipedia) is way more cohesive than a speech from a podcast (+ ASR errors).

To further make the training set similar to the inference domain, we preprocessed the training set to remove unknown words (instead of masking them) and stop words. The removal of stop words deteriorates cohesiveness, but as was mentioned, since we don't take it into account anymore, this shouldn't make any difference. The two transformers where able to find topic boundaries even with no stop words, as stop words don't contribute topic change detection. Note that in inference time, similar preprocessing needs to be done for the ASR output.


### Reference implementation: [EducationalTestingService/CATS](https://github.com/EducationalTestingService/CATS)
