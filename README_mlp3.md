# makemore_mlp3.ipynb

## Overview
This Jupyter notebook implements a hierarchical, deep character-level neural network for name generation, building on previous makemore MLP projects. It features custom neural network layers, batch normalization, and a hierarchical architecture using consecutive flattening, inspired by Andrej Karpathy's makemore series. The notebook demonstrates advanced PyTorch concepts and efficient training for character-level language modeling.

## Main Steps
1. **Data Loading**: Reads a list of names from `names.txt`.
2. **Vocabulary Building**: Creates mappings from characters to indices and vice versa, including a special end-of-word token.
3. **Dataset Preparation**: Converts names into input-output pairs for training, using a larger context window (block size).
4. **Model Definition**: Implements a hierarchical deep neural network using custom Embedding, FlattenConsecutive, Linear, BatchNorm1d, and Tanh layers, assembled in a sequential architecture.
5. **Training**: Trains the model using cross-entropy loss and manual gradient descent, with learning rate scheduling and batch normalization.
6. **Visualization**: Plots training loss and provides code for evaluating model performance.
7. **Sampling**: Generates new names by sampling from the trained model, character by character.

## Key Concepts and Logic
- **Hierarchical Architecture**: The model uses consecutive flattening to build hierarchical representations, allowing deeper and more expressive networks.
- **Custom Layers**: Defines its own Embedding, Linear, BatchNorm1d, Tanh, and FlattenConsecutive classes to illustrate internal workings of neural network layers.
- **Batch Normalization**: Normalizes activations within each layer, improving training stability and convergence.
- **Manual Training Loop**: Training is performed with explicit forward and backward passes, parameter updates, and learning rate scheduling.
- **Sampling**: The model generates names by iteratively predicting the next character given a context window, until the end-of-word token is produced.

## How to Use
1. Ensure you have `names.txt` in the same directory as the notebook.
2. Install required packages: `torch`, `matplotlib`, and `numpy`.
3. Run all cells in order to train the model and generate new names.

## File Structure
- `makemore_mlp3.ipynb`: This notebook (hierarchical deep MLP model)
- `names.txt`: List of names for training

## Comparison with Previous MLP Models

| Model         | Architecture         | Context Size | Custom Layers | BatchNorm | Hierarchical/Flattening | Training Stability | Expressiveness |
|---------------|---------------------|--------------|---------------|-----------|------------------------|-------------------|---------------|
| MLP (makemore_mlp)   | Shallow MLP (1-2 layers) | Small (3)     | No            | No        | No                     | Basic             | Limited        |
| MLP2 (makemore_mlp2) | Deep MLP (many layers)   | Small (3)     | Yes           | Yes       | No                     | Improved          | Better         |
| MLP3 (makemore_mlp3) | Hierarchical Deep MLP    | Large (8)     | Yes           | Yes       | Yes                    | Best              | Highest        |

**Key Improvements in makemore_mlp3.ipynb:**
- **Hierarchical Architecture:** Uses consecutive flattening to build hierarchical representations, enabling deeper and more expressive models.
- **Larger Context Window:** Increases block size, allowing the model to capture longer dependencies in names.
- **Custom Layers:** All layers (Embedding, Linear, BatchNorm1d, Tanh, FlattenConsecutive) are implemented from scratch for educational clarity and flexibility.
- **Batch Normalization:** Present in both MLP2 and MLP3, but MLP3 leverages it in a hierarchical structure for even greater stability.
- **Training Stability and Expressiveness:** MLP3 is more robust to hyperparameters and can learn more complex patterns, resulting in better generalization and name generation quality.

This notebook demonstrates the evolution from a simple MLP to a deep, hierarchical model, showing how architectural innovations and improved training techniques lead to better performance in character-level language modeling.

## References
- [Andrej Karpathy's makemore series](https://github.com/karpathy/makemore)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---
This notebook is for educational purposes and demonstrates hierarchical deep neural network concepts, custom layer implementation, and batch normalization for character-level language modeling.
