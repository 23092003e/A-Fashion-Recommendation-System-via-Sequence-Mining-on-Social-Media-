import numpy as np
import pandas as pd
import emot
from pathlib import Path

def read_data(path: str) -> pd.DataFrame:
    """
    Read a JSON files and returns a pandas DataFrame
    
    Args: path(str): path to the JSON file
    
    Return: pd.DataFrame: DataFrame containing the data from the JSON file
    """
    try:
        return pd.read_json(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {path}")
    except ValueError as e:
        raise ValueError(f"Error parsing JSON file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error reading file: {str(e)}")

def replace_emojis_with_text(text: str) -> str:
    """
    This function replaces emojis with their textual representation
    
    Args: text(str): text containing emojis
    
    Return: str: text with emojis replaced by their textual representation
    """
    emot_obj = emot.emot()
    
    try:
        emoji_info = emot_obj.emoji(text)
        num_emojis = len(emoji_info["value"])
        
        for i in range(num_emojis):
            text = text.replace(emoji_info["value"][i], emoji_info["mean"][i])
            
    except Exception as e:
        print(f"An Error occurred while processing the text: {text}. The error is as follows: {e}")
        
    return text

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the dataframe and returns a new dataframe.
    
    Args:
    df (pd.DataFrame): Dataframe to be processed.
    
    Returns:
    pd.DataFrame: A processed pandas dataframe.
    """
    
    # Selecting relevant columns
    df = df[["id", "type", "commentsCount", "likesCount", "latestComments", "images"]]
    
    # Rename columns
    new_columns = {
        "id": "id",
        "commentsCount": "n_comments",
        "likesCount": "n_likes",
        "latestComments": "comments",
        "images": "image"
    }
    df = df.rename(columns=new_columns)
    
    # Filtering out rows where type is "Video", likes count is -1.0 and image is not null 
    df = df[(df["type"] != "Video")] 
    df = df[df["n_likes"] != -1.0]
    df = df[df["image"].notna()]
    
    #Removing rows with no image:
    df = df[df["image"].apply(len) > 0]
    
    #Selecting the first image if there are multiple images
    df["image"] = df["image"].apply(lambda x: x[0])
    
    #Extracting text from comments:
    df["comments"] = df["comments"].apply(lambda x: [i["text"] for i in x if "text" in i])
    
    # Reseting the id
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index + 1
    
    # Converting empty lists in comments to np.nan and creating a separate dataframe for comment
    df["comments"] = df["comments"].apply(lambda x: x if isinstance(x, list) and x else np.nan)
    df_comments = df.explode("comments")[["id", "comments"]]
    
    # Replace emojis in comments with their text descriptions
    df_comments["comments"] = df_comments["comments"].apply(replace_emojis_with_text)
    
    # Removing comments columns from the original dataframe
    df = df.drop("comments", axis=1)

    return df, df_comments    

def save_data(df: pd.DataFrame, df_comments: pd.DataFrame) -> None:
    """
    Saves the dataframe and comments dataframe as csv files.
    
    Args:
    df (pd.DataFrame): Dataframe to be saved.
    df_comments (pd.DataFrame): Comments dataframe to be saved.
    """
    output_dir = Path("C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    posts_path = output_dir / "posts.csv"
    comments_path = output_dir / "posts_comments.csv"
    
    df.to_csv(posts_path, index=False)
    df_comments.to_csv(comments_path, index=False, sep=';')
    print(f"Data saved successfully to {output_dir}")

def main():
    try:
        # Define input paths
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
        
        print("Data processing completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()