import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import numpy as np
import warnings as w
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

w.filterwarnings('ignore')

def load_data(file):
    data = pd.read_csv(file)
    return shuffle(data)

def plot_graphs(data):
    graphs = {
        1: ('Class', 'Marks Class Count Graph'),
        2: ('Semester', 'Marks Class Semester-wise Graph'),
        3: ('gender', 'Marks Class Gender-wise Graph'),
        4: ('NationalITy', 'Marks Class Nationality-wise Graph'),
        5: ('GradeID', 'Marks Class Grade-wise Graph'),
        6: ('SectionID', 'Marks Class Section-wise Graph'),
        7: ('Topic', 'Marks Class Topic-wise Graph'),
        8: ('StageID', 'Marks Class Stage-wise Graph'),
        9: ('StudentAbsenceDays', 'Marks Class Absent Days-wise Graph')
    }
    while True:
        print("\n".join([f"{key}. {value[1]}" for key, value in graphs.items()]) + "\n10. Exit")
        ch = int(input("Enter Choice: "))
        if ch == 10:
            print("Exiting...")
            break
        elif ch in graphs:
            print(f"Loading {graphs[ch][1]}...")
            t.sleep(1)
            sb.countplot(x=graphs[ch][0], hue='Class', data=data, hue_order=['L', 'M', 'H'])
            plt.show()

def preprocess_data(data):
    drop_cols = ["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction", "ParentAnsweringSurvey", "AnnouncementsView"]
    data.drop(columns=drop_cols, inplace=True)
    
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = LabelEncoder().fit_transform(data[column])
    
    return data

def train_models(feats_train, lbls_train):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Perceptron': Perceptron(),
        'Logistic Regression': LogisticRegression(),
        'MLP Classifier': MLPClassifier(activation='logistic')
    }
    for name, model in models.items():
        model.fit(feats_train, lbls_train)
    return models

def evaluate_models(models, feats_test, lbls_test):
    for name, model in models.items():
        lbls_pred = model.predict(feats_test)
        acc = (lbls_pred == lbls_test).mean()
        print(f"\n{name} Accuracy: {round(acc, 3)}")
        print(classification_report(lbls_test, lbls_pred))
        t.sleep(1)

data = load_data("AI-Data.csv")
plot_graphs(data)
data = preprocess_data(data)

ind = int(len(data) * 0.70)
feats, lbls = data.values[:, :-1], data.values[:, -1]
feats_train, feats_test = feats[:ind], feats[ind:]
lbls_train, lbls_test = lbls[:ind], lbls[ind:]

models = train_models(feats_train, lbls_train)
evaluate_models(models, feats_test, lbls_test)

print("Exiting...")
