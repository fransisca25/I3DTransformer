# I3D-Transformer Model
## Model Architecture
![image](https://github.com/fransisca25/I3DTransformer/blob/master/i3dtransformer_architecture.png)

## Overview
The I3D-Transformer model is an advanced neural network designed for spatiotemporal video analysis. It combines the power of Inflated 3D ConvNet (I3D) for feature extraction with a Transformer architecture to enhance temporal dynamics and feature representation. This model is particularly suited for tasks like action recognition in videos. For this research we are using sign language videos.

## Key Components
### Inception I3D:
- Extracts spatiotemporal features from video frames using 3D convolutions.
- Captures both spatial and temporal patterns in video sequences.

### Transformer:
- Processes the embeddings produced by I3D using multi-head attention mechanisms.
- Enhances the learning of temporal relationships between different segments of the video.

### Multi-Layer Perceptron (MLP):

- Refines the features extracted by the Transformer.
- Produces logits for classification and embeddings for further analysis.

## Research Status
The I3D-Transformer model is currently under research. We are actively working on evaluating its performance, optimizing its components, and exploring its potential applications. Feedback and contributions are welcome as we continue to refine and improve the model.
