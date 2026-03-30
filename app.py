import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="my streamlit app", page_icon="🚀", layout="wide")
st.title("my first streamlit app")
st.markdown("_ _ _")
st.header("welcome to my app")
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name} 👋")
    if st.button("Click me"):
        st.success("Button clicked!")
        import pandas as pd

df = pd.DataFrame({"Numbers": [1, 2, 3, 4], "Squares": [1, 4, 9, 16]})

st.dataframe(df)
