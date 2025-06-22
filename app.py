# Step 1: Install libraries
!pip install -q tensorflow pandas scikit-learn

# Step 2: Import libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 3: Create simple dataset
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

# Step 4: Preprocessing
X = df['title'].values
y = df['category'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, padding='post', maxlen=10)

X_train, X_test, y_train, y_test = train_test_split(padded, y_encoded, test_size=0.2)

# Step 5: Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 7: Prediction example
def predict_category(title):
    seq = tokenizer.texts_to_sequences([title])
    padded_seq = pad_sequences(seq, padding='post', maxlen=10)
    pred = model.predict(padded_seq)
    category = le.inverse_transform([pred.argmax()])
    return category[0]

print(predict_category("Advanced Human Anatomy"))
