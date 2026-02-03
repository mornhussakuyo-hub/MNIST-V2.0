# MNIST Handwritten Digit Recognition Project

## Project Overview

This project implements a two-layer neural network for handwritten digit recognition using the MNIST dataset. The model is built from scratch in Python with NumPy, focusing on educational clarity and practical usage. It includes complete workflows for data preprocessing, model training, prediction, and visualization.

**Note**: This project is developed by non-native English speakers. Please use standard English when submitting issues or queries to facilitate translation and communication.

## System Requirements

- **Operating System**: Linux only
- **Python**: 3.8 or higher (managed via Conda environment)
- **Memory**: Minimum 4GB RAM (8GB recommended for full dataset training)
- **Storage**: At least 1GB free space for data and models

## Project Structure

```
/mnist_project/
├── data/                   # Directory for dataset files (CSV format)
├── models/                 # Saved trained models (.npz files)
├── predicts/              # Prediction output files
├── results/               # Training history files
├── scripts/               # All Python scripts
│   ├── common.py          # Common utilities (path validation)
│   ├── model.py           # Neural network model definition
│   ├── predict.py         # Prediction script
│   ├── train.py           # Training script
│   ├── utils.py           # Data processing utilities
│   └── visualize.py       # Training visualization script
├── visualize/             # Generated visualization images
└── environment.yml        # Conda environment configuration
```

## Installation Guide

### 1. Clone the Repository

```
git clone git@github.com:mornhussakuyo-hub/MNIST-V2.0.git
cd MNIST-V2.0
```

### 2. Set Up Conda Environment

This project uses Conda for dependency management. Create the environment using the provided configuration file:

```
conda env create -f environment.yml
```

Activate the environment:

```
conda activate mnist_env
```

**Expected Dependencies** (as defined in environment.yml):

- Python 3.8+
- numpy
- pandas
- matplotlib
- pathlib
- argparse

If the environment.yml file is missing or incomplete, you can install dependencies manually:

```
pip install numpy pandas matplotlib
```

## Data Preparation

### Data Format Requirements

- All data must be in **CSV format**
- First column must contain labels (0-9)
- Subsequent 784 columns represent 28x28 pixel values (0-255)
- Header row is optional but not required

### Expected File Structure

```
/data/
├── mnist_train.csv    # Training data (60,000 samples)
└── mnist_test.csv     # Test data (10,000 samples)
```

### Obtaining MNIST Data

You can download pre-formatted MNIST CSV files from:[Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), or our google-drive [link](https://drive.google.com/file/d/1CQlq1BwaSJzXFdFafM-gVlOQLYMrIIxx/view?usp=sharing).

## Usage Guide

### 1. Obtaining pre-trained model

We provide a pre-trained model. The training commands are as follows:

```
python train.py --model-name pre_train_model
```

You can download it by clicking this [link](https://drive.google.com/file/d/1qTkiMAyv6ZEfFGjfyN1xKXGOsSMcqt-L/view?usp=sharing), which should download a `pre_train_model.npz` file. You can use this model in conjunction with the predict.py script.

For the train history of this model, click this [link](https://drive.google.com/file/d/1jQevvDM_goYX8iKk2AqWYM5XJcMEeuSU/view?usp=sharing)

### 2. Training the Model

The training script offers extensive customization options for hyperparameters and training configuration.

**Basic Training Command:**

```
cd scripts
python train.py --model-name my_model --epochs 50 --batch-size 32 --learning-rate 0.01
```

**Complete Training Example with All Options:**

```
python train.py \
    --model-name advanced_model \
    --traindt ../data/mnist_train.csv \
    --testdt ../data/mnist_test.csv \
    --model-dir ../models \
    --results-dir ../results \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.01 \
    --hidden-size 256 \
    --val-ratio 0.2 \
    --usage-ratio 0.8 \
    --reg-rate 0.001 \
    --LR-dc on \
    --LR-dc-num 20
```

**Parameter Explanations:**

- `--model-name`: Unique identifier for your model (required)
- `--traindt`: Path to training data CSV (default: ../data/mnist_train.csv)
- `--testdt`: Path to test data CSV (default: ../data/mnist_test.csv)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Mini-batch size for training (default: 32)
- `--learning-rate`: Initial learning rate (default: 0.01)
- `--hidden-size`: Number of neurons in hidden layer (default: 128)
- `--val-ratio`: Proportion of training data for validation (default: 0.2)
- `--usage-ratio`: Proportion of full dataset to use (default: 1.0)
- `--reg-rate`: L2 regularization strength (default: 0.0)
- `--LR-dc`: Learning rate decrease ("on"/"off", default: "off")
- `--LR-dc-num`: Epoch interval for LR decrease (default: 20)

**Expected Output:**

```
Process start...
Network: 784 -> 128 -> 10
Hyperparameter: LR=0.01, Epochs=50, Batch=32
Learning rate decrease was off.
Regularization rate was 0.0.

Start loading data...
Training progress with loss/accuracy metrics per epoch...
Test data acc: 97.50%
```

### 3. Making Predictions

Use trained models to predict digits on new data.

**Basic Prediction Command:**

```
python predict.py --data ../data/mnist_test.csv --model ../models/my_first_model.npz
```

**Advanced Prediction with Custom Output:**

```
python predict.py \
    --data ../data/mnist_test.csv \
    --model ../models/my_first_model.npz \
    --outdir ../predicts
```

**Parameter Explanations:**

- `--data`: Path to test data CSV file (required)
- `--model`: Path to trained model .npz file (required)
- `--outdir`: Output directory for predictions (default: ../predicts)

**Output Features:**

- Accuracy calculation if labels are present
- Detailed per-sample predictions
- Error handling with informative messages

**Example Output File** (`../predicts/my_first_model_predicts.out`):

```
Model: my_first_model
Test data: ../data/mnist_test.csv
Number of sample: 10000
The accuracy on this data-set was 97.50%

Sample      1: Predicted=7, True=Yes
Sample      2: Predicted=2, True=Yes
...
```

### 4. Visualizing Training Progress

Generate plots to analyze model training performance.

**Visualization Command:**

```
python visualize.py --hist ../results/my_first_model_training_history.npz
```

**Generated Visualizations:**

- Training/validation loss curves
- Training/validation accuracy curves
- Combined overview plots
- High-resolution PNG images (300 DPI)

**Output Location:**

- Visualizations saved to `../visualize/my_first_model_visualize/`

## Advanced Features

### Weight Initialization Methods

The model supports three initialization strategies (set in model.py):

- **He initialization**: Recommended for ReLU activation (default)
- **Xavier initialization**: Suitable for tanh/sigmoid
- **Normal initialization**: Simple random weights

### Regularization Options

- L2 regularization via `--reg-rate`parameter
- Early stopping capability through validation monitoring
- Data usage ratio for training subset selection

### Learning Rate Scheduling

- Automatic learning rate reduction every N epochs
- Configurable reduction factor (current implementation halves LR)
- Manual learning rate adjustment capabilities

## Model Architecture Details

**Network Structure:**

- Input layer: 784 neurons (28×28 pixels)
- Hidden layer: Configurable size (128 default), ReLU activation
- Output layer: 10 neurons (digits 0-9), Softmax activation
- Loss function: Categorical cross-entropy
- Optimization: Mini-batch gradient descent

**Key Implementation Features:**

- Efficient vectorized operations using NumPy
- Proper gradient checking and numerical stability
- Comprehensive caching for intermediate results
- Memory-efficient batch processing

## Troubleshooting Common Issues

### File Path Errors

```
# Ensure you're in the scripts directory
cd /mnist_project/scripts

# Use absolute paths if relative paths fail
python predict.py --data /full/path/to/mnist_project/data/mnist_test.csv
```

### Memory Issues

- Reduce batch size (`--batch-size 16`)
- Use data subset (`--usage-ratio 0.5`)
- Close other memory-intensive applications

### Performance Optimization

- For faster training, increase batch size
- For better accuracy, increase hidden layer size
- Use learning rate scheduling for stable convergence

### Data Format Issues

```
# Verify CSV format quickly
import pandas as pd
df = pd.read_csv('your_data.csv')
print(f"Shape: {df.shape}, Columns: {df.columns[:5]}")
```

## Output Files Description

### Model Files (.npz)

- Weight matrices (W1, W2)
- Bias vectors (b1, b2)
- Model metadata and configuration

### Result Files (.npz)

- Training loss history
- Validation loss history
- Accuracy metrics per epoch

### Prediction Files (.out)

- Model information and test details
- Per-sample predictions with confidence
- Accuracy statistics for labeled data

## Contributing Guidelines

We welcome contributions! Please note:

1. **Use Standard English** for all communications, issues, and pull requests
2. **Follow the existing code style** and structure
3. **Test changes thoroughly** before submitting
4. **Update documentation** to reflect modifications
5. **Add comments** for complex logic sections

### Development Setup

```
git clone [repository-url]
cd mnist_project
conda env create -f environment.yml
conda activate mnist_env
# Implement your changes...
python -m pytest tests/  # If test suite exists
```

## License and Citation

This project is provided for educational and research purposes. Please cite the original MNIST dataset and this implementation if used in academic work.

## Support and Contact

For technical support:

1. Check this README and code comments first
2. Search existing GitHub issues
3. Create a new issue with detailed error description and system information

## Future Enhancements

Potential improvements for future versions:

- CNN architecture support
- Better encapsulation
- Implementation of Convolutional and Pooling Layers
- Real-time training monitoring
- Hyperparameter optimization
- Docker containerization
- Web interface for predictions
- Additional dataset support