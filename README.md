# Epsilon Optimizer

This tool calculates the optimal epsilon value that maximizes the welfare function for differential privacy parameters.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you're in the epsilon_optimizer directory:
```bash
cd cs208-tool/epsilon_optimizer
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and go to:
```
http://127.0.0.1:5001
```

## How It Works

The tool calculates the optimal epsilon value that maximizes the welfare function:

\[
W = \int_{\epsilon_a}^{\epsilon_b} \left( \frac{1}{x} \right) \left( 1 - \frac{1}{N \epsilon} \right) \int_{\max(\epsilon, \epsilon_a)}^{\epsilon_b} \frac{1}{y} \, dy \, dx + N \int_{\epsilon}^{\epsilon_b} \left( 1 - \frac{\epsilon}{x} \right) \, dx
\]

Where:
- εₐ (epsilon_a) is the minimum epsilon value
- εᵦ (epsilon_b) is the maximum epsilon value
- N is the number of subjects in the study

## Parameters

- **Minimum Epsilon (εₐ)**: The lower bound of the epsilon distribution (default: 0.01)
- **Maximum Epsilon (εᵦ)**: The upper bound of the epsilon distribution (default: 2.0)
- **Number of Subjects (N)**: The number of participants in the study (default: 100) 