import streamlit as st
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer
import scipy
import openai
import pandas as pd
import tiktoken
from write_title import write_title
from write_text import  write_maintext

df = pd.read_csv("df/ShibuyaKabanAi_referenceText - Output.csv")
openai.api_key = st.text_input("openAI APIkey", "")
df['emb'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

if 'df' not in st.session_state:
    st.session_state['df'] = df
if 'ai_text' not in st.session_state:
    st.session_state['ai_text'] = 'waiting'

def titlegen():
    # Create a text input
    user_inputtext = st.text_input("text", "革製品は使えば使うほど洗練されていく")
    # Create a button
    send_button = st.button("Gen")
    # Handle button click event
    if send_button:
        # Perform some action with the user input
        ai_res = process_input_title(user_inputtext)
        st.write(ai_res)


def process_input_title(user_inputtext):
    # Do something with the user input
    ai_res = write_title(df, keyword=user_inputtext)
    return ai_res

def process_input_text(user_title):
    # Do something with the user input
    ai_text = write_maintext(df, query=user_title)
    st.session_state['ai_text'] =ai_text
    return ai_text

st.title("Title Generator")
titlegen()

st.title("Text Generator")
def textgen():
    # Create a text input
    user_title = st.text_input("text", "copy and paste your title")
    # Create a button
    gentext_button = st.button("Gen text")
    # Handle button click event
    if gentext_button:
        # Perform some action with the user input
        ai_text = process_input_text(user_title)
        st.write(ai_text)
textgen()

def predict(text):
    # Load BERT model and tokenizer
    model_name = 'yukaru1986/SK_class'
    model_token = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token="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")
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_token)

    # Tokenize input text
    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Make prediction
    outputs = model(**encoded_input)
    temp = outputs.logits.tolist()
    percent = scipy.special.softmax(temp)
    zero = round(percent[0][0],2)
    one = round(percent[0][1],2)
    predictions = outputs.logits.argmax(dim=1)

    # Return prediction
    return predictions.item(),zero,one


def classification(ai_text):
    st.title("BERT Classification App")

    # Get user input
    user_input = st.text_area("Enter a sentence for classification:",ai_text)

    # Make prediction when user submits the form
    if st.button("Classify"):
        prediction = predict(user_input)
        if prediction[0] == 1:
            kekka = "成功する可能性が高いです。"
        else:
            kekka = "再考したほうが良いです。"
        st.subheader(f"このテキストは、{kekka}")
        st.write(f"成功投稿である確率: {prediction[2]}")
        st.write(f"失敗投稿の確率: {prediction[1]}")
classification(ai_text="waiting")
