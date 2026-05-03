# src/controlled_generation.py
# IMPROVEMENT: Generate 100 per batch x 4 batches = 400 per label + counterfactuals (~500 total)
import os, time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])

def generate_controlled(label_name, n_per_batch=100, n_batches=4):
    all_samples = []
    for batch_num in range(n_batches):
        prompt = f'''Generate {n_per_batch} short movie review sentences with the following rules:
1. Sentiment: {label_name}
2. Keep demographic mentions minimal and neutral — avoid he/she/him/her where possible
3. When characters must be mentioned, use exactly equal male and female references
4. Vary sentence structure: some long, some short, some with questions
5. Vary vocabulary: do not repeat the same adjectives
6. Include diverse names and contexts (different genres, settings)
7. Format: one review per line, no numbering, no extra text

Return only the reviews, one per line.'''

        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.75,
            max_tokens=2000,
        )
        lines = [l.strip() for l in response.choices[0].message.content.split('\n')
                 if len(l.strip()) > 10]
        all_samples.extend(lines[:n_per_batch])
        time.sleep(0.5)
        print(f'  Batch {batch_num+1}/{n_batches} done ({len(all_samples)} samples so far)')
    return all_samples

def generate_counterfactuals(base_samples, batch_size=20):
    '''Generate gender-swapped versions of samples in batches'''
    all_cf = []
    for i in range(0, min(len(base_samples), 60), batch_size):
        batch = base_samples[i:i+batch_size]
        prompt = '''For each sentence below, create an alternative version that:
- Keeps the SAME meaning and sentiment
- Changes all male references (he/him/his/man) to female (she/her/hers/woman) and vice versa
- Changes male names (John, David, Mark) to female names (Sarah, Emma, Lisa) and vice versa
Output only the modified sentences, one per line, same order as input.

Input sentences:
''' + '\n'.join(batch)

        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        lines = [l.strip() for l in response.choices[0].message.content.split('\n')
                 if len(l.strip()) > 10]
        all_cf.extend(lines[:batch_size])
        time.sleep(0.5)
    return all_cf

print('Generating controlled positive samples...')
controlled_pos = generate_controlled('positive', n_per_batch=100, n_batches=4)
print('Generating controlled negative samples...')
controlled_neg = generate_controlled('negative', n_per_batch=100, n_batches=4)

print('Generating counterfactual pairs...')
cf_pos = generate_counterfactuals(controlled_pos[:60])
cf_neg = generate_counterfactuals(controlled_neg[:60])

all_controlled = (
    [(s, 1, 'controlled') for s in controlled_pos + cf_pos] +
    [(s, 0, 'controlled') for s in controlled_neg + cf_neg]
)
ctrl_df = pd.DataFrame(all_controlled, columns=['sentence', 'label', 'source'])
ctrl_df.to_csv('data/synthetic/sst2_controlled.csv', index=False)
print(f'\nGenerated {len(ctrl_df)} controlled samples')
print(f'Label distribution: {ctrl_df.label.value_counts().to_dict()}')