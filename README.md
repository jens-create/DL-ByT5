# DL-ByT5
Deep learning project within NLP using the ByT5 model to grammatical error correction.


The ByT5 has been fine-tuned with a sentence-2-sentence approach based on Google pre-trained ByT5 small model. [https://huggingface.co/google/byt5-small]

The so-called multilexnorm model is based on the work of [https://huggingface.co/ufal/byt5-small-multilexnorm2021-da]. It is also based on the small ByT5 but it has already been fine-tuned on twitter data with a masked-word-2-word approach. In this project we further fine-tune the model.

The data that we have fine-tuned with are not public, hence we can only upload a limited part of the data set - it corresponds to the test set such that the main results can be reproduced.

We use the word-error-rate, BLEU and GLEU scores as performance metrics.

We have gathered our main findings in two jupyter notebooks, each corresponding to an approach. They should be viewed as demos. For the "working" code we refer to the folders named "approach"_individual_files. The fine-tuning, inferring and evaluation of the models is located in these folders.

sentence-2-sentence approach:
- See ByT5.ipynb for training, inferring, evaluation and visualisation of the model.

masked-word-2-word approach:
- See MLN.ipynb for training, inferring and evaluation.


Contact us at s183568@student.dtu.dk or s193713@student.dtu.dk
