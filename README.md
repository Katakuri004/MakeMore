MakeMore: Character-Level Neural Network Name Generation
=======================================================

## Project Overview
This project is a hands-on exploration of character-level neural network language models for generating names, inspired by Andrej Karpathy's makemore series. It consists of a series of Jupyter notebooks that progressively build from a simple MLP to deep and hierarchical architectures, with custom layer implementations, batch normalization, and detailed training visualizations. The project is educational and demonstrates core deep learning concepts in PyTorch for character-level modeling.

### Notebooks Included
- **makemore_mlp.ipynb**: Simple MLP for name generation
- **makemore_mlp2.ipynb**: Deep MLP with custom layers and batch normalization
- **makemore_mlp3.ipynb**: Hierarchical deep MLP with advanced architecture

## Results
- The models successfully learn to generate plausible, novel names character by character after training on a dataset of real names.
- Training and analysis plots (see below and in the assets folder) show effective learning, stable gradients, and well-behaved activations.
- Hierarchical and deep models (MLP2, MLP3) demonstrate improved generalization and more realistic name generation compared to the basic MLP.
- Custom layer implementations and batch normalization enable stable training even for very deep networks.

Example generated names (from the best model):

```
Marlia
Jorine
Kallie
Zan
Lynnia
Ryn
Tayden
Sallia
Brin
Liora
```

