# Heart Disease Prediction

This repository contains a machine learning project designed to predict the presence of heart disease in patients based on key health indicators. By analyzing medical data and applying classification algorithms, this project demonstrates how data science can aid in early detection and treatment of heart diseases.

## Features

- **Data Exploration and Cleaning**: Identifies and handles duplicate or missing data to ensure high-quality input.
- **Visual Analysis**: Provides bubble plots, heatmaps, and other visualizations to uncover patterns and correlations among health metrics.
- **Machine Learning Models**: Implements multiple models to predict heart disease.
- **Model Optimization**: Applies cross-validation and hyperparameter tuning to improve accuracy and reliability.

## Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Tools and Libraries

- Python
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn, Plotly (visualization)
- Scikit-learn (machine learning)
- Bubbly (interactive visualizations)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`heart.csv`) in the project directory.

## Usage

1. Run the notebook to preprocess the data and train models:
   ```bash
   jupyter notebook Prediction_of_heart_disease_using_machine_learning.ipynb
   ```

2. Visualize the relationships between key features like blood pressure, cholesterol, and age.

3. Evaluate and compare the performance of different machine learning models.

4. Use the best-performing model for predictions.

## Dataset

The dataset should be a CSV file (`heart.csv`) containing key health metrics such as:
- Age
- Sex
- Cholesterol
- Resting Blood Pressure
- Max Heart Rate
- Target (0: No Heart Disease, 1: Heart Disease)

## Results

- The best model achieved an accuracy of **XX%** (replace with actual results).
- Detailed evaluation metrics such as precision, recall, and F1-score are included.

## Future Work

- Incorporate additional features for improved predictions.
- Deploy the model using Flask or Streamlit for real-time diagnosis.
- Explore deep learning methods for enhanced accuracy.

## Contribution

Contributions are welcome! Fork this repository and submit pull requests to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
