import pandas as pd
import numpy as np
from pathlib import Path
import emot

# Read data from JSON file
def read_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
    except ValueError as e:
        raise ValueError(f"Error parsing JSON file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error reading file: {str(e)}")

# Function to replace emojis with text
def replace_emojis_with_text(text: str) -> str:
    emot_obj = emot.emot()
    try:
        emoji_info = emot_obj.emoji(text)
        num_emojis = len(emoji_info["value"])
        for i in range(num_emojis):
            text = text.replace(emoji_info["value"][i], emoji_info["mean"][i])
    except Exception as e:
        print(f"An error occurred while processing the text: {text}. The error is as follows: {e}")
    return text

# Function to process DataFrame and separate DataFrame for comments
def process_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    # Select necessary columns
    selected_columns = ["id", "timestamp", "ownerUsername", "caption", "hashtags", 
                        "likesCount", "commentsCount", "latestComments", "images"]
    df = df[selected_columns]
    
    # Rename columns for better understanding
    df = df.rename(columns ={
        "id": "post_id",
        "timestamp": "timestamp",
        "ownerUsername": "ownerUsername",
        "caption": "caption",
        "hashtags": "hashtags",
        "likesCount": "likesCount",
        "commentsCount": "commentsCount",
        "latestComments": "comments",
        "images": "image"
    })
    
    # Filter out rows with likesCount = -1.0 and no images
    df = df[df["likesCount"] != -1.0]
    df = df[df["image"].notna()]
    # Remove rows with empty image list
    df = df[df["image"].apply(len) > 0]
    
    # Reset index and create post_id column with numbering from 1
    df.reset_index(drop=True, inplace=True)
    df["post_id"] = df.index + 1
    
    # If there are multiple images, select the first image
    df["image"] = df["image"].apply(lambda x: x[0])
    
    # Process caption: fill empty values and replace newline characters with spaces
    df["caption"] = df["caption"].fillna(" ").str.replace("\n", " ")
    
    # Process hashtags: if it's a list, join it into a string, otherwise assign an empty string
    df["hashtags"] = df["hashtags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    
    # Split comment data
    comments_data = []
    for _, row in df.iterrows():
        pid = row["post_id"]
        latest_comments = row["comments"]
        if isinstance(latest_comments, list) and len(latest_comments) > 0:
            for comment in latest_comments:
                comment_text = comment.get("text", "")
                comment_owner = comment.get("ownerUsername", "")
                comment_timestamp = comment.get("timestamp", "")
                comment_likes = comment.get("likesCount", 0)
                comments_data.append({
                    "post_id": pid,
                    "ownerUsername": comment_owner,
                    "timestamp": comment_timestamp,
                    "comments": comment_text,
                    "likes": comment_likes
                })
        else:
            # If there are no comments, you can skip or add a row with the value np.nan
            comments_data.append({
                "post_id": pid,
                "ownerUsername": np.nan,
                "timestamp": np.nan,
                "comments": np.nan,
                "likes": np.nan
            })
    df_comments = pd.DataFrame(comments_data)
    
    # Replace emojis in comments
    if not df_comments.empty:
        df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)
        # Process timestamp for comments
        df_comments["timestamp"] = pd.to_datetime(df_comments["timestamp"], errors='coerce')
    
    # Process timestamp for posts
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    
    # Process likesCount and commentsCount: fill with 0 if missing
    df["likesCount"] = df["likesCount"].fillna(0).astype(int)
    df["commentsCount"] = df["commentsCount"].fillna(0).astype(int)
    
    # After splitting comments, remove the comments column from the main df
    df = df.drop("comments", axis=1)
    
    return df, df_comments

# Function to save data to CSV file
def save_data(df: pd.DataFrame, df_comments: pd.DataFrame) -> None:
    output_dir = Path("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    posts_path = output_dir / "posts.csv"
    comments_path = output_dir / "posts_comments.csv"
    
    df.to_csv(posts_path, index=False, encoding="utf-8-sig")
    df_comments.to_csv(comments_path, index=False, encoding="utf-8-sig")
    
    print(f"Data saved successfully to {output_dir}")

def main():
    try:
        # Define the path to the raw data directory
        input_dir = Path("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/raw")
        path1 = input_dir / "posts_1.json"
        path2 = input_dir / "posts_2.json"
        
        print(f"Reading data from {input_dir}")
        
        # Read data from 2 JSON files
        df_1 = read_data(path1)
        df_2 = read_data(path2)
        
        # Concatenate data from 2 files
        df = pd.concat([df_1, df_2])
        df, df_comments = process_data(df)
        save_data(df, df_comments)
        
        print("Data processing completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
