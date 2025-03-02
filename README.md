# CIFAR-10 Feedforward Neural Network

## Overview
This project implements a feedforward neural network trained on the CIFAR-10 dataset using backpropagation. Various optimizers, activation functions, and hyperparameters are tested to determine the best-performing model.

## Dataset
- **CIFAR-10**: A dataset of 60,000 color images in 10 classes, with 6,000 images per class.
- Images are of size 32x32 pixels.

## Requirements
Ensure you have the following dependencies installed:
```
pip install torch torchvision matplotlib numpy scikit-learn seaborn
```

## Training the Model
To train the model, run:
```
python train.py --epochs 10 --optimizer adam --batch_size 32 --learning_rate 0.001 --hidden_layers 128 64
```

### Available Hyperparameters:
- `--epochs`: Number of training epochs (default: 10)
- `--optimizer`: Choose from ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam']
- `--batch_size`: Set batch size (default: 32)
- `--learning_rate`: Set learning rate (default: 0.001)
- `--hidden_layers`: Define number of neurons per hidden layer (e.g., 128 64)

## Evaluating the Model
To test the model on the test dataset:
```
python evaluate.py --model best_model.pth
```

## Results & Analysis
- Validation accuracy is recorded for different hyperparameter settings.
- A confusion matrix is generated for the best-performing model.
- The comparison between Cross-Entropy and Mean Squared Error loss is documented.

## Recommendations for MNIST
Based on CIFAR-10 experiments, the following configurations are recommended for MNIST:
1. **Config 1:** 3 layers, 64 neurons per layer, Adam optimizer, LR = 0.001  
2. **Config 2:** 4 layers, 128 neurons per layer, SGD optimizer, LR = 0.0001  
3. **Config 3:** 5 layers, 32 neurons per layer, RMSprop optimizer, LR = 0.001  

## License
This project is for educational purposes.
