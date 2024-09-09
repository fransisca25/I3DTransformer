# I3D-Transformer Model
## Overview
The I3D-Transformer model is an advanced neural network designed for spatiotemporal video analysis. It combines the power of Inflated 3D ConvNet (I3D) for feature extraction with a Transformer architecture to enhance temporal dynamics and feature representation. This model is particularly suited for tasks like action recognition in videos. For this research we are using sign language videos.

## Model Architecture
![image](https://github.com/fransisca25/I3DTransformer/blob/master/i3dtransformer_architecture.png)

## Model Architecture Details
### I3D Backbone
The Inflated 3D ConvNet (I3D) serves as the backbone for spatiotemporal feature extraction. It processes video frames to extract features with a dimensionality of 1024. Instead of using additional linear layers typically added before the Transformer, this model directly reshapes the I3D features into a shape of 16 x 64 to align with the Transformer's input requirements. This choice helps to reduce model complexity while preserving essential spatiotemporal information.

### Transformer Model
The reshaped features are fed into a Transformer encoder, which utilizes self-attention mechanisms to capture contextual relationships among the feature elements. The encoded features are then passed to the Transformer decoder, where a special "Class Query" (a learnable, randomly initialized parameter) guides the decoding process. This class query acts as a unique template, enhancing the model's ability to recognize specific signs.

### MLP Head
The output from the Transformer decoder is reshaped back to 1024 dimensions and processed by an MLP head. This head includes a residual block, which maintains the 1024-dimensional features, and an adaptor module with two linear layers and Leaky ReLU activations. The dimensions are transformed from 1024 to 512, and then from 512 to 256. A dropout layer is included to prevent overfitting. The final layer produces logits used for classification, with the Cross-Entropy loss function incorporating the Softmax function.

## Key Components
### Inception I3D:
- Extracts spatiotemporal features from video frames using 3D convolutions.
- Captures both spatial and temporal patterns in video sequences.
- I3D model is based on the work described in: <br> <b><a href="https://arxiv.org/pdf/1705.07750">Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset by Joao Carreira and Andrew Zisserman.</a></b>

### Transformer:
- Processes the embeddings produced by I3D using multi-head attention mechanisms.
- Enhances the learning of temporal relationships between different segments of the video.
- Transformer model is based on the work described in: <br> <b><a href="https://arxiv.org/pdf/1706.03762">Attention is All You Need by Vaswani et al.</a></b>

### Multi-Layer Perceptron (MLP):
- Refines the features extracted by the Transformer.
- Produces logits for classification and embeddings for further analysis.

## Research Status
The I3D-Transformer model is currently under research. We are actively working on evaluating its performance, optimizing its components, and exploring its potential applications. Feedback and contributions are welcome as we continue to refine and improve the model.
