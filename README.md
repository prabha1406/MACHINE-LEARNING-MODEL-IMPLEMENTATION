# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY:CODTECH IT SOLUTIONS

NAME: Mohana Prabha k

INTERN ID: CT04DH2091

DOMAIN: PYTHON PROGRAMMING

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH

DESCRIPTION:

Project Title:

Spam Detection Using Machine Learning – CodTech Internship Task 4

Project Description (Point Format):

This project was developed as part of Task 4 of the CodTech IT Solutions Internship Program.

The objective was to build a predictive machine learning model using scikit-learn that can classify SMS messages as either Spam or Ham (not spam).

The dataset used in this project contains SMS messages labeled as 'spam' or 'ham', sourced from a public dataset hosted online in .tsv format.

The project uses Python and popular machine learning libraries such as:

pandas for data handling

scikit-learn for model training and evaluation

CountVectorizer for text vectorization

seaborn and matplotlib for data visualization

The entire implementation was done using Google Colab, which allows easy execution and sharing of Jupyter notebooks.

Workflow / Steps Followed:

Importing Libraries: All necessary libraries were imported including pandas, numpy, seaborn, matplotlib, and various modules from sklearn.

Loading Dataset: The dataset was read directly from a URL using pd.read_csv() with tab separation.

Exploratory Data Analysis (EDA):

The first few rows of the data were displayed to understand its structure.

Basic info was extracted using df.info() and df.head().

Label Encoding:
The categorical column ‘label’ was mapped to numeric values using:

ham → 0

spam → 1

Train-Test Split:
The dataset was split into training and testing sets using an 80-20 split ratio with train_test_split.

Text Preprocessing:
CountVectorizer was used to convert the text messages into numeric format using a bag-of-words model.

Model Building:
A Multinomial Naive Bayes classifier was chosen due to its effectiveness in text classification tasks.

The model was trained using model.fit() on the training data.

Prediction:
Predictions were made on the test dataset using the model.predict() method.

Model Evaluation:
The model was evaluated using:

accuracy_score

confusion_matrix

classification_report

These metrics help measure the performance of the model across precision, recall, and f1-score.

Visualization:
A Seaborn countplot was created to visualize the number of spam vs ham messages in the dataset.

Results:

The model achieved an accuracy above 97%, demonstrating high effectiveness in detecting spam messages.

The confusion matrix showed very few false positives or false negatives.

The classification report indicated strong precision and recall, especially for the spam class.

Conclusion:

This project demonstrates a complete text classification pipeline, from loading raw data to building and evaluating a machine learning model.

It showcases the power of Naive Bayes in Natural Language Processing (NLP) tasks like spam detection.

The notebook includes all code, outputs, and visualizations for easy understanding and reproducibility.

The final version is suitable for uploading to GitHub or submitting as part of an internship deliverable.

OUTPUT:

![Image](https://github.com/user-attachments/assets/485858fe-419a-4bd6-a3f5-d4c0675e8fa6)

![Image](https://github.com/user-attachments/assets/cfab6093-2934-43b8-a05c-6364d0c302f1)
