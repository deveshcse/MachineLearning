# Import necessary libraries
import gensim
import pandas as pd

# Read the CSV file containing India news headlines using pandas
df = pd.read_csv("dataset/india-news-headlines.csv")

# Extract only the 'headline_text' column from the DataFrame
df = df['headline_text']

# Apply simple preprocessing to the headlines using gensim's utility function
text = df.apply(gensim.utils.simple_preprocess)

# Create a Word2Vec model with specified parameters
model = gensim.models.Word2Vec(
    window=10,        # The maximum distance between the current and predicted word within a sentence
    min_count=2,       # Ignores all words with a total frequency lower than this
    workers=4,         # Number of CPU cores to use for training the model
)

# Build the vocabulary of the model from the preprocessed headlines
model.build_vocab(text, progress_per=1000)

# Train the Word2Vec model on the preprocessed headlines
model.train(text, total_examples=model.corpus_count, epochs=model.epochs)

# Save the trained Word2Vec model to a file
model.save("./News_Embeddings_gensim.model")

# Find the most similar words to the word "bad" in the trained model's vocabulary
model.wv.most_similar("bad")

# Calculate the similarity between the words "great" and "good" in the trained model's vocabulary
model.wv.similarity(w1="great", w2="good")


# Load the saved Word2Vec model from the file
model = gensim.models.Word2Vec.load("./News_Embeddings_gensim.model")

# Find the most similar words to the word "bad" in the trained model's vocabulary
similar_words = model.wv.most_similar("bad")
print("Most similar words to 'bad':", similar_words)

# Calculate the similarity between the words "great" and "good" in the trained model's vocabulary
similarity_score = model.wv.similarity(w1="great", w2="good")
print("Similarity between 'great' and 'good':", similarity_score)