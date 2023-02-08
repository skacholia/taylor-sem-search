import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import tiktoken
import streamlit as st
import streamlit.components.v1 as components

st.title("Find an applicable Taylor Swift Lyric")
st.markdown(
    "This app uses OpenAI's embeddings model to find the most relevant Taylor Swift lyric for any given description"
)

lyrics = pd.read_csv("tswift_embed.csv")

openai.api_key = st.secrets["openaiKey"]
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(embedding_encoding)

embedding_model = "text-embedding-ada-002"
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_embed(df, description, n=3, pprint=True):
    description_embedding = get_embedding(
        description,
        model="text-embedding-ada-002"
    )
    df["similarity"] = df.ada_embedding.apply(lambda x: cosine_similarity(x, description_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

title = st.text_input('Description', "I've got so many haters")
st.write(search_embed(lyrics, title, n=3))

