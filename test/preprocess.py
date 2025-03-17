import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
import unicodedata
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

comments = r'C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\data\processed\posts_comments.csv'
posts = r'C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\data\processed\posts.csv'

# Step 1: Read and clean the data
def read_and_clean_data():
    # Read CSV files with encoding that supports emoji
    try:
        posts_df = pd.read_csv(posts, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            posts_df = pd.read_csv(posts, encoding='utf-8-sig')
        except UnicodeDecodeError:
            # Try latin-1 as a fallback for extreme cases
            posts_df = pd.read_csv(posts, encoding='latin-1')
        
    try:
        comments_df = pd.read_csv(comments, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            comments_df = pd.read_csv(comments, encoding='utf-8-sig')
        except UnicodeDecodeError:
            # Try latin-1 as a fallback for extreme cases
            comments_df = pd.read_csv(comments, encoding='latin-1')
    
    # Clean posts data
    posts_df = posts_df.drop(columns=['image'])
    posts_df.rename(columns={'ownerUsername': 'postUser'}, inplace=True)

    posts_df['caption'] = posts_df['caption'].fillna('')
    posts_df['hashtags'] = posts_df['hashtags'].fillna('')
    
    # Clean comments data
    comments_df['comments'] = comments_df['comments'].fillna('')
    
    return posts_df, comments_df

# Step 2: Improved text preprocessing function with emoji handling
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    try:
        # Handle emoji by either:
        # Option 1: Replace emoji with their textual descriptions
        text = emoji.demojize(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        
        # Handle unicode characters and normalize
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        # If any error occurs, return an empty string or handle it safely
        print(f"Error processing text: {str(e)[:100]}...")
        return ''

# Step 3: Extract hashtags as a list (with error handling)
def extract_hashtags(hashtag_string):
    if not isinstance(hashtag_string, str):
        return []
    
    try:
        # Handle emoji in hashtags
        hashtag_string = emoji.demojize(hashtag_string)
        
        # Split hashtags by common separators
        hashtags = re.split(r'[,;#\s]+', hashtag_string)
        
        # Remove empty strings
        hashtags = [tag for tag in hashtags if tag]
        
        return hashtags
    except Exception as e:
        print(f"Error extracting hashtags: {str(e)[:100]}...")
        return []

# Step 4: Combine post and comments data with better error handling
def combine_post_comments(posts_df, comments_df):
    try:
        # Group comments by post_id
        comments_grouped = comments_df.groupby('post_id')['comments'].apply(
            lambda x: ' '.join([str(comment) for comment in x if pd.notna(comment)])
        ).reset_index()
        
        # Merge with posts data
        combined_df = posts_df.merge(comments_grouped, on='post_id', how='left')
        combined_df['comments'] = combined_df['comments'].fillna('')
        
        # Apply preprocessing with error handling
        print("Preprocessing captions...")
        combined_df['cleaned_caption'] = combined_df['caption'].apply(preprocess_text)
        
        print("Preprocessing comments...")
        combined_df['cleaned_comments'] = combined_df['comments'].apply(preprocess_text)
        
        print("Extracting hashtags...")
        combined_df['hashtag_list'] = combined_df['hashtags'].apply(extract_hashtags)
        
        # Create consolidated text
        combined_df['consolidated_text'] = combined_df['cleaned_caption'] + ' ' + combined_df['cleaned_comments'] + ' ' + combined_df['hashtag_list'].apply(lambda x: ' '.join(x))
        
        return combined_df
    except Exception as e:
        print(f"Error in combining data: {str(e)}")
        raise

# Step 5: Create embeddings using Word2Vec (alternative to BERT)
def create_word2vec_embeddings(combined_df):
    try:
        # Tokenize the consolidated text
        print("Tokenizing text for Word2Vec...")
        tokenized_texts = []
        
        for text in combined_df['consolidated_text']:
            if isinstance(text, str) and text.strip():
                tokens = [word for word in word_tokenize(text.lower()) if word.isalpha()]
                tokenized_texts.append(tokens)
            else:
                tokenized_texts.append([])
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        w2v_model = Word2Vec(sentences=tokenized_texts,
                            vector_size=200,  # Size of embedding vector
                            window=5,         # Context window size
                            min_count=1,      # Ignore words with frequency < min_count
                            workers=4)        # Number of processor cores
        
        # Create document vectors by averaging word vectors
        print("Creating document vectors...")
        doc_vectors = []
        
        for tokens in tokenized_texts:
            if tokens:
                # Get vectors for words in the model's vocabulary
                word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
                if word_vectors:
                    # Average the word vectors
                    doc_vector = np.mean(word_vectors, axis=0)
                else:
                    # If no words are in vocabulary, use zeros
                    doc_vector = np.zeros(w2v_model.vector_size)
            else:
                # Empty document gets zero vector
                doc_vector = np.zeros(w2v_model.vector_size)
                
            doc_vectors.append(doc_vector)
        
        # Add vectors to dataframe
        combined_df['embedding'] = doc_vectors
        
        # Save the Word2Vec model for future use
        w2v_model.save('instagram_word2vec.model')
        print("Saved Word2Vec model to 'instagram_word2vec.model'")
        
        return combined_df, w2v_model
    except Exception as e:
        print(f"Error in creating Word2Vec embeddings: {str(e)}")
        # Return dataframe without embeddings if there's an error
        combined_df['embedding'] = [np.zeros(200) for _ in range(len(combined_df))]
        return combined_df, None

# Step 6: Alternative method - Create TF-IDF vectors
def create_tfidf_vectors(combined_df):
    try:
        # Ensure all texts are valid strings
        combined_df['consolidated_text_safe'] = combined_df['consolidated_text'].apply(
            lambda x: x if isinstance(x, str) and x.strip() else " "
        )
        
        # Create TF-IDF vectors
        print("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(max_features=500)  # Limit features to avoid memory issues
        tfidf_matrix = vectorizer.fit_transform(combined_df['consolidated_text_safe'])
        
        # Convert sparse matrix to dense array for easier handling
        tfidf_dense = tfidf_matrix.toarray()
        
        # Add vectors to dataframe
        combined_df['tfidf_vector'] = list(tfidf_dense)
        
        # Save the vectorizer for future use
        import pickle
        with open('instagram_tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        print("Saved TF-IDF vectorizer to 'instagram_tfidf_vectorizer.pkl'")
        
        return combined_df, vectorizer
    except Exception as e:
        print(f"Error in creating TF-IDF vectors: {str(e)}")
        # Return dataframe without vectors if there's an error
        combined_df['tfidf_vector'] = [np.zeros(10) for _ in range(len(combined_df))]
        return combined_df, None

# Step 7: Apply attention mechanism (using TF-IDF scores)
def apply_attention(combined_df, vectorizer=None):
    try:
        if vectorizer is None:
            # Create new vectorizer if not provided
            vectorizer = TfidfVectorizer(max_features=10000)
            tfidf_matrix = vectorizer.fit_transform(combined_df['consolidated_text'].fillna(''))
        else:
            # Use the provided vectorizer
            tfidf_matrix = vectorizer.transform(combined_df['consolidated_text'].fillna(''))
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get important words for each post
        print("Extracting important words...")
        important_words_list = []
        
        for i, row in combined_df.iterrows():
            try:
                # Get the TF-IDF scores for this document
                doc_tfidf = tfidf_matrix[i]
                
                # Get the top N words
                top_n = 5
                sorted_indices = np.argsort(doc_tfidf.toarray().flatten())[::-1]
                top_indices = sorted_indices[:top_n]
                important_words = [feature_names[i] for i in top_indices if i < len(feature_names)]
                
                important_words_list.append(important_words)
            except Exception as e:
                print(f"Error extracting important words for post {i}: {str(e)[:100]}...")
                important_words_list.append([])
        
        combined_df['important_words'] = important_words_list
        
        return combined_df
    except Exception as e:
        print(f"Error in applying attention: {str(e)}")
        # Return the dataframe without attention if there's an error
        combined_df['important_words'] = [[] for _ in range(len(combined_df))]
        return combined_df

# Step 8: Main function to execute the pipeline with both embedding methods
def process_instagram_data():
    try:
        # Read and clean data
        posts_df, comments_df = read_and_clean_data()
        print(f"Read {len(posts_df)} posts and {len(comments_df)} comments")
        
        # Combine post and comments data
        combined_df = combine_post_comments(posts_df, comments_df)
        print(f"Combined data shape: {combined_df.shape}")
        
        # Use both embedding methods
        print("\n--- Creating Word2Vec Embeddings ---")
        combined_df, w2v_model = create_word2vec_embeddings(combined_df)
        print("Created Word2Vec embeddings for all posts")
        
        print("\n--- Creating TF-IDF Vectors ---")
        combined_df, vectorizer = create_tfidf_vectors(combined_df)
        print("Created TF-IDF vectors for all posts")
        
        # Apply attention mechanism using TF-IDF
        combined_df = apply_attention(combined_df, vectorizer)
        print("Applied attention mechanism to identify important words")
        
        # Save the processed data
        # Save only the necessary columns to avoid memory issues with large vectors
        save_columns = [
            'post_id', 'timestamp', 'postUser', 'caption', 'hashtags', 
            'likesCount', 'commentsCount', 'important_words', 'consolidated_text'
        ]
        
        # Save embeddings separately
        embedding_df = combined_df[['post_id', 'embedding']]
        embedding_df.to_pickle('instagram_word2vec_embeddings.pkl')
        print("Saved Word2Vec embeddings to 'instagram_word2vec_embeddings.pkl'")
        
        tfidf_df = combined_df[['post_id', 'tfidf_vector']]
        tfidf_df.to_pickle('instagram_tfidf_vectors.pkl')
        print("Saved TF-IDF vectors to 'instagram_tfidf_vectors.pkl'")
        
        # Save the rest of the data
        combined_df[save_columns].to_csv('processed_instagram_data.csv', index=False)
        print("Saved processed data to 'processed_instagram_data.csv'")
        
        return combined_df
    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        raise

# Execute the pipeline
if __name__ == "__main__":
    try:
        processed_data = process_instagram_data()
        
        # Display sample results
        sample = processed_data[['post_id', 'postUser', 'consolidated_text', 'important_words']].head(3)
        print("\nSample Results:")
        print(sample)
    except Exception as e:
        print(f"Error executing the pipeline: {str(e)}")