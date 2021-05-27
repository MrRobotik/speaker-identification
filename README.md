# Speaker identification
<b>X-vector</b> DNN architecture for speech embeddings (512 dim. vector),<br> which can be used for speaker identification or verification.

implemented in python3 using:
* PyTorch for neural network building and training
* librosa for audio manipulation and feature extraction
* Bob for evaluation of biometric performance metric

The project consists of 6 mini-apps:
* `extract_feats.py` which precomputes mel-spectrograms and MFCC's for each utterance
* `train_nn_model.py` which trains the DNN model using selected loss function and optimizer
* `calc_embeds.py` precomputes the embeddings using trained x-vector model (for evaluation)
* `test_model.py` evaluates the model by finding EER and plotting DET curve
* `embed_plots.py` plots t-SNE projection of the embeddings (visual evaluation)
* `demo.py` demo application which demonstrates speaker comparison in real-time using mic
