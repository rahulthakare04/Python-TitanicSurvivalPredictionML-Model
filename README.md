# Titanic Survival Prediction

This project predicts the survival of passengers on the Titanic using a Decision Tree Classifier. It uses the Titanic dataset and performs data preprocessing to handle missing values and encode categorical data.

---

## Project Structure

### Data Preprocessing and Feature Engineering
- **01SurvivalPredictor.py**: 
  - Loads the Titanic dataset.
  - Drops irrelevant columns and handles missing values.
  - Encodes the 'Sex' column into numerical values.
  - Trains a Decision Tree Classifier model to predict survival.
  - Makes predictions using hardcoded input samples.

### Model Training and Evaluation
- **02SurvivalPredictorAccuracy.py** and **03SurvivalPredictorAccuracy.py**: 
  - Loads and preprocesses the dataset.
  - Splits the data into training and testing sets.
  - Trains and tests the model 10 times using a Decision Tree Classifier.
  - Calculates and displays the average accuracy score of the model.

### Model Saving and Deployment
- **04TrainAndSaveModel.py**: 
  - Trains a Decision Tree Classifier model on the entire dataset.
  - Saves the trained model using `joblib`.

- **05UseSavedModel.py**: 
  - Loads a pre-trained model using `joblib`.
  - Makes predictions using the saved model.


- pandas
- scikit-learn
- joblib

You can install these using:
```bash
pip install pandas scikit-learn joblib
