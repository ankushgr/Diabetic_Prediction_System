Diabetes prediction using Random Forest and Decision Tree

The following project is machine learning project to forecast if a person is diabetic or non-diabetic with 2 classification model -Random Forest and Decision Tree. The dataset being used is the Pima Indians Diabetes Dataset, which is commonly used to predict medical outcomes.

🔍 Project Overview

The program:

Reads diabetes.data and processes it. csv`.

Does feature scaling via StandardScaler.

Trains two models: Random Forest and Tree.

-Compares both models with 5-fold cross-validation and test accuracy.

Offers ability for users to submit their own data and get predictions from both models.

📁 Dataset

Name the dataset diabetes.  and stored in the same location as the script. It has the following columns:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

(0 = NoDiabetic or 1 = Diabetic) available as Outcome

⚙️ Requirements

Make sure you have the following libraries in your system:

pip install pandas numpy scikit-learn

📊 Model Configuration

Random Forest Parameters:

n_estimators=100

max_depth=10

min_samples_split=5

min_samples_leaf=4

max_features='sqrt'

Decision Tree Parameters:

max_depth=10

min_samples_split=5

min_samples_leaf=4

📈 Evaluation

Both the models are described with:

Accuracy Score

5-Fold Cross Validation

🧠 Sample Input Features

Respond with the following when prompted for values of:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age
