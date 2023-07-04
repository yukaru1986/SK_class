import openai

from create_context import create_context

def write_title(
        df,
        model="gpt-3.5-turbo",
        keyword="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=500,
        stop_sequence=None
):
    context = create_context(
        keyword,
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
                      {"role": "user", "content": f"You are a marketing professional.\
                      Make 5 title ideas for sns post.\
                      Keyword is {query}.\
                      Output in Japanese based on the context below,\
                      The target audience consists of individuals, particularly creators and artists.\
                      Please use a formal writing style.\
                      Keep sentences short.\
                      The length of the title sholud be 10 to 15 characters.\
                      Output in Japanese.\
                      reference some of the context below.\
                      and if the request can't be answered based on the context, say I don't know\n\nContext: {context}"}],
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
