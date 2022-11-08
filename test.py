import pickle
import streamlit as st

st.title("Which author do you write like?")

with open ("models/author_pipe.pkl", "rb") as pickle_in:
    pipe = pickle.load(pickle_in)
    
your_text = st.text_input("Please enter some text:", max_chars=500)

predicted_auther = pipe.predict([your_text])[0]

st.write(f"You write like {predicted_auther.title()}.")