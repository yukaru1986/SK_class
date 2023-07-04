import streamlit as st
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer
import scipy
import openai

import pandas as pd

from write_title import write_title

df = pd.read_csv("df/shibuya_df.csv")
openai.api_key = st.text_input("openAI APIkey", "")

def titlegen():
    st.title("Title Generator")
    # Create a text input
    user_inputtext = st.text_input("text", "革製品は使えば使うほど洗練されていく")
    # Create a button
    send_button = st.button("Gen")
    # Handle button click event
    if send_button:
        # Perform some action with the user input
        process_input(user_inputtext)

def process_input(user_inputtext):
    # Do something with the user input
    ai_res = write_title(df, keyword=user_inputtext)
    st.write(ai_res)

titlegen()

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


def main():
    st.title("BERT Classification App")

    # Get user input
    user_input = st.text_area("Enter a sentence for classification:")

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


if __name__ == '__main__':
    main()
