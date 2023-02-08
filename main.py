import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import tiktoken
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import urllib.request
from PIL import Image

urllib.request.urlretrieve("https://en.wikipedia.org/wiki/Fearless_%28Taylor%27s_Version%29#/media/File:Fearless_(Taylor's_Version)_(2021_album_cover)_by_Taylor_Swift.png", 'taylor.png')
img = Image.open('taylor.png')

st.title("✨ There's a Taylor Swift lyric for that ✨")
st.markdown(
    "Put in a description or phrase, and this app will find an applicable Taylor Swift lyric to match it."
)

lyrics = pd.read_csv("https://media.githubusercontent.com/media/skacholia/taylor-sem-search/main/tswift_embed.csv")
lyrics["ada_embedding"] = lyrics["ada_embedding"].apply(lambda x: np.fromstring(x.strip("[").strip("]"), sep=","))

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

description = st.text_input('Description', "Yeongseo is my friend")
st.button(label = "Find a lyric",
          on_click = st.write(search_embed(lyrics, description, n=3)))
st.image(img)


