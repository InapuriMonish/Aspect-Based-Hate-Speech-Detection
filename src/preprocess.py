import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

def load_data(path="../data/train.csv"):
    df = pd.read_csv(path)
    df = df[["comment_text"] + LABEL_COLS]
    df["comment_text"] = df["comment_text"].astype(str)
    return df

def split_data(df, test_size=0.1):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )
    return train_df, val_df

if __name__ == "__main__":
    df = load_data()
    train_df, val_df = split_data(df)
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
