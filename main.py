import streamlit as st
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer


def predict(text):
    # Load BERT model and tokenizer
    model_name = 'yukaru1986/SK_class'
    model_token = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    model = AutoModelForSequenceClassification.from_pretrained(model_name,use_auth_token="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")
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
    predictions = outputs.logits.argmax(dim=1)

    # Return prediction
    return predictions.item()


# Create Streamlit app
def main():
    st.title("BERT Classification App")

    # Get user input
    user_input = st.text_input("Enter a sentence for classification:")

    # Make prediction when user submits the form
    if st.button("Classify"):
        prediction = predict(user_input)
        st.write(f"Predicted class: {prediction}")


if __name__ == '__main__':
    main()
