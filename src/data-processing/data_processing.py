import pandas as pd
import numpy as np
from pathlib import Path
import emot

# Read Data from json files
def read_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
    except ValueError as e:
        raise ValueError(f"Error parsing JSON file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error reading file: {str(e)}")
    
# Replace emojis with their textual representation
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

# Processes DataFrame and returns new dataframe.
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Selecting relevant columns
    selected_columns = ["id", "timestamp", "ownerUsername", "type", "caption", "hashtags", "likesCount", "commentsCount", "latestComments", "images"]
    df = df[selected_columns]
    
    # Rename columns
    df = df.rename(columns ={
        "id": "post_id",
        "timestamp": "timestamp",
        "ownerUsername": "ownerUsername",
        "type": "type",
        "caption": "caption",
        "hashtags": "hashtags",
        "likesCount": "likesCount",
        "commentsCount": "commentsCount",
        "latestComments": "comments",
        "images": "image"
    })
    
    # Filtering out rows where type is "Video", like count is -1.0 and image is not null
    df = df[(df["type"] != "Video")] 
    df = df[df["likesCount"] != -1.0]
    df = df[df["image"].notna()]

    # Removing rows with no image:
    df = df[df["image"].apply(len) > 0]
    
    #Resetting index
    df.reset_index(drop=True, inplace=True)
    df["post_id"] = df.index + 1
    
    # Selecting only the first image if there are multiple images
    df["image"] = df["image"].apply(lambda x: x[0])
    
    # Process caption
    df["caption"] = df["caption"].fillna("").str.replace("\n", " ")
    
    # Process hashtags
    df["hashtags"] = df["hashtags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    
    # Extracting text from comments
    df["comments"] = df["comments"].apply(lambda x: [i["text"] for i in x if "text" in i])
    # Converting empty lists in comments to np.nan and creating a separate dataframe for comments
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) and x else np.nan)
    df_comments = df.explode("comments")[["post_id", "comments"]]
    
    # Replace emojis in comments with their text descriptions
    df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)

    # Process timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    
    # Process likesCount and commentsCount
    df["likesCount"] = df["likesCount"].fillna(0).astype(int)
    df["commentsCount"] = df["commentsCount"].fillna(0).astype(int)
        
    # Ordering by timestamp
    df.sort_values(by=["timestamp"], inplace=True)
    
    df = df.drop("comments", axis=1)
    
    return df, df_comments

# Save the dataframe as csv file
def save_data(df: pd.DataFrame, df_comments: pd.DataFrame) -> None:
    output_dir = Path("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    posts_path = output_dir / "posts.csv"
    comments_path = output_dir / "posts_comments.csv"
    
    df.to_csv(posts_path, index=False)
    df_comments.to_csv(comments_path, index=False)
    
    print(f"Data saved successfully to {output_dir}")
    

def main():
    try:
        # Define input data
        input_dir = Path("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/raw")
        path1 = input_dir / "posts_1.json"
        path2 = input_dir / "posts_2.json"
        
        print(f"Reading data from {input_dir}")
        
        # Read, process and save the data
        df_1 = read_data(path1)
        df_2 = read_data(path2)
        
        df = pd.concat([df_1, df_2])
        df, df_comments = process_data(df)
        save_data(df, df_comments)
        
        print("Data processing completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()