# Step 1: Install & Import Libraries
!pip install -q tensorflow pandas scikit-learn

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Load Sample Data (you can replace with your own CSV)
data = {
    'title': [
        'Essentials of Human Anatomy',
        'Textbook of Medical Physiology',
        'Pharmacology Made Easy',
        'Fundamentals of Pathology',
        'Medical Microbiology Basics'
    ],
    'category': ['Anatomy', 'Physiology', 'Pharmacology', 'Pathology', 'Microbiology']
}
df = pd.DataFrame(data)

# Step 3: Preprocessing
X = df['title'].values
y = df['category'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, padding='post', maxlen=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded, y_encoded, test_size=0.2)

# Step 4: Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 6: Test
def predict_category(title):
    seq = tokenizer.texts_to_sequences([title])
    padded_seq = pad_sequences(seq, padding='post', maxlen=10)
    pred = model.predict(padded_seq)
    category = le.inverse_transform([pred.argmax()])
    return category[0]

# Example
print(predict_category("Advanced Human Anatomy"))
