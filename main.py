import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import tiktoken
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image

img = Image.open("taylor.png")

@st.cache_data
def read_lyric_csv():
   return pd.read_csv("https://media.githubusercontent.com/media/skacholia/taylor-sem-search/main/tswift_embed.csv")

lyrics = read_lyric_csv();
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
        
    result_text = '\n\n'.join([f'**{r[:r.index(":")]}:** {r[r.index(":")+2:]}' for r in results])
    return result_text

st.title("✨ There's a Taylor lyric for that ✨")
st.markdown(
 "Put in a description or phrase, and this app will find applicable Taylor Swift lyrics to match it."
)
st.markdown(
 "**For example**:"
)

st.markdown(
 "\"Melancholic romance\" → \"What a sad, beautiful, tragic love affair\""
)

st.markdown(
   "\"I'm coding at the library and drinking a latte\" → \"Coffee at midnight\""
)
   

description = st.text_area('Description', "")

if st.button(label = "Find a lyric"):
    try:
        result = search_embed(lyrics, description, n=3)
        st.write(result)
    except:
        st.write("An error occurred while processing your request.")

st.image(img, use_column_width=True)
