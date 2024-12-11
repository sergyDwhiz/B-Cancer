# Cancer Prediction Project

This project is designed to predict cancer outcomes using a pre-trained machine learning model. The project reads data from a CSV file, processes it, and uses the model to make predictions.

## Project Structure

- **data/**: Contains the dataset used for predictions.
  - `data.csv`: The CSV file containing the data.
- **src/**: Contains the source code and model for predictions.
  - `model.pkl`: The pre-trained machine learning model.
  - `predictor.ipynb`: The Jupyter notebook for data processing and prediction.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sergyDwhiz/cancer-prediction.git
    cd cancer-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Open the Jupyter notebook:
    ```sh
    jupyter notebook src/predictor.ipynb
    ```

2. Follow the instructions in the notebook to load the data, process it, and make predictions using the pre-trained model.

### Example

Here is an example of how to use the notebook to make predictions:

```python
import pandas as pd
import joblib

# Load the data
df = pd.read_csv('../data/data.csv')

# Load the pre-trained model
model = joblib.load('model.pkl')

# Make predictions
y_pred = model.predict(df)
print(y_pred)
```

### Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.