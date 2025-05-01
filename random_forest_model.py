import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
df = pd.read_csv("final_protein_dataset.csv")

label_encoder = LabelEncoder()
df['Label_encoded'] = label_encoder.fit_transform(df['Label'])

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['Sequence'])
sequences = tokenizer.texts_to_sequences(df['Sequence'])
X = pad_sequences(sequences, maxlen=50)
y = to_categorical(df['Label_encoded'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, np.argmax(y_train, axis=1))

# Evaluate
y_pred = rf.predict(X_test)
y_true = np.argmax(y_test, axis=1)

print("Random Forest Evaluation:")

# Convert label_encoder.classes_ to strings for classification_report
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()