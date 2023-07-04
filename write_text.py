import openai

from create_context import create_context

def write_maintext(
        df,
        model="gpt-3.5-turbo",
        query="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=450,
        stop_sequence=None
):
    context = create_context(
        query,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = openai.ChatCompletion.create(
            messages=[{"role": "system",
                       "content": "You are a marketing professional."},
                      {"role": "user", "content": f"\
                      Write a article.\
                      The title is {query}.\
                      The goal of this article is to increase awareness of the brand.\
                      The target audience consists of young individuals, particularly creators and artists.\
                      Please use a casual writing style.\
                      Keep Sentences short.\
                      The desired length for the article is between 300 and 400 characters.\
                      Output in Japanese.\
                      reference some of the context below.\
                      If the request can't be answered based on the context, say \"I don't know\"\n\nContext: {context}"}],
            temperature=0.3,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )

        return response["choices"][0]['message']['content']
    except Exception as e:
        print(e)
        return ""
