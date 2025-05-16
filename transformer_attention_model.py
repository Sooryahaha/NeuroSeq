import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer, Add
from tensorflow.keras.callbacks import LearningRateScheduler

# Load dataset
df = pd.read_csv("final_protein_dataset.csv")
label_encoder = LabelEncoder()
df["Label_encoded"] = label_encoder.fit_transform(df["Label"])

# Tokenize sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df["Sequence"])
sequences = tokenizer.texts_to_sequences(df["Sequence"])
max_len = 100
X = pad_sequences(sequences, maxlen=max_len, padding="post")
y = to_categorical(df["Label_encoded"])

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Positional Encoding
def positional_encoding(length, depth):
    depth = depth // 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

# Attention Pooling Layer
class AttentionPooling(Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()
    def build(self, input_shape):
        self.dense = Dense(1)
    def call(self, inputs):
        weights = tf.nn.softmax(self.dense(inputs), axis=1)
        output = tf.reduce_sum(inputs * weights, axis=1)
        return output

# Build model
vocab_size = len(tokenizer.word_index) + 1
num_classes = y.shape[1]
embed_dim = 256

inp = Input(shape=(max_len,))
x = Embedding(vocab_size, embed_dim)(inp)

# Add positional encoding
pos_encoding = positional_encoding(max_len, embed_dim)
x = x + pos_encoding

# Stack Transformer Blocks
for _ in range(3):
    x = transformer_encoder(x, head_size=64, num_heads=6, ff_dim=256, dropout=0.2)

x = AttentionPooling()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=inp, outputs=out)

# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch > 10:
        return lr * 0.5
    return lr

lr_callback = LearningRateScheduler(scheduler)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=32, callbacks=[lr_callback])

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTransformer Model Accuracy: {acc * 100:.2f}%")

# Predictions and metrics
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_.astype(str)))

# Confusion matrix heatmap
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Transformer Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# F1 score bar plot
f1_scores = f1_score(y_true_classes, y_pred_classes, average=None)
plt.figure(figsize=(8, 5))
sns.barplot(x=label_encoder.classes_, y=f1_scores, palette="viridis")
plt.title("F1 Score per Class")
plt.ylabel("F1 Score")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Final metrics summary
summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [
        accuracy_score(y_true_classes, y_pred_classes),
        precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0),
        recall_score(y_true_classes, y_pred_classes, average='weighted'),
        f1_score(y_true_classes, y_pred_classes, average='weighted')
    ]
})
print("\nFinal Summary Table:")
print(summary_df.to_markdown(index=False))

# Risk/confidence scatter plot
risk_scores = np.max(y_pred, axis=1)
plt.figure(figsize=(8, 5))
plt.scatter(range(len(risk_scores)), risk_scores, c=risk_scores, cmap='coolwarm', alpha=0.6)
plt.title("Prediction Confidence per Sample")
plt.xlabel("Sample Index")
plt.ylabel("Confidence")
plt.colorbar(label="Confidence Score")
plt.tight_layout()
plt.show()