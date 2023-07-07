import numpy as np
import openai

def cosine_similarity(a, b):
    # openai.embeddings_utils.cosine_similarity()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_context(
        question, df, max_len=1800, size="ada"
):
    # Get the embeddings for the question
    q_embeddings = \
    openai.Embedding.create(input=question, engine='text-embedding-ada-002')[
        'data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = df['emb'].apply(lambda x: cosine_similarity(q_embeddings, x))

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)
