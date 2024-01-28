
import pandas as pd
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# Load the dataset (assuming it's a CSV file with a 'headline_text' column)
df = pd.read_csv('dataset/india-news-headlines.csv')

# Vocabulary size
voc_size = 10000

# Use one-hot encoding to convert text to numerical vectors
df['one_hot_encoding'] = [one_hot(text, voc_size) for text in df['headline_text']]

# Display the DataFrame with one-hot encoded headlines
print(df.head())

# Set the maximum sentence length for padding
sent_length = 12

# Pad sequences to ensure uniform length
embedded_docs = pad_sequences(df['one_hot_encoding'], padding='pre', maxlen=sent_length)
print(embedded_docs)

# Define the size of the embedded vectors
embedded_vector_size = 5

# Create a Sequential model
model = Sequential()

# Add an Embedding layer with vocabulary size, embedding vector size, and input length
model.add(Embedding(voc_size, embedded_vector_size, input_length=sent_length, name="embedding"))

# Compile the model using 'adam' optimizer and 'mse' loss
model.compile(optimizer='adam', loss='mse')

# Display the model summary
model.summary()

# Print the embedding vector for the first document in the dataset
print(model.predict(embedded_docs[0:1]))
