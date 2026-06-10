# src/controlled_generation.py
import os, time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from config import PROMPT_POSITIVE, PROMPT_NEGATIVE, DOMAIN, CONTROLLED_PATH

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])

os.makedirs('data/synthetic', exist_ok=True)

def generate_controlled(condition_description, n_per_batch=100, n_batches=4):
    all_samples = []
    for batch_num in range(n_batches):
        prompt = f'''Generate {n_per_batch} short {DOMAIN} sentences with these rules:
1. Topic/condition: {condition_description}
2. Balance demographic mentions equally — equal male and female subjects
3. Vary sentence structure: some long, some short, some questions
4. Vary vocabulary: avoid repeating the same words
5. Include diverse names, ages, and backgrounds
6. Format: one sentence per line, no numbering, no extra text

Return only the sentences.'''
        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.75, max_tokens=2000,
        )
        lines = [l.strip() for l in response.choices[0].message.content.split('\n') if len(l.strip()) > 10]
        all_samples.extend(lines[:n_per_batch])
        time.sleep(0.5)
        print(f'  Batch {batch_num+1}/{n_batches} done ({len(all_samples)} so far)')
    return all_samples

def generate_counterfactuals(base_samples, batch_size=20):
    all_cf = []
    for i in range(0, min(len(base_samples), 60), batch_size):
        batch = base_samples[i:i+batch_size]
        prompt = f'''For each {DOMAIN} sentence below, create an alternative version that:
- Keeps the SAME meaning and sentiment
- Swaps all male references (he/him/his/man/male/father/husband) to female equivalents and vice versa
- Swaps male names (John, David, Mark) to female names (Sarah, Emma, Lisa) and vice versa
Output only the modified sentences, one per line, same order.

Input:
''' + '\n'.join(batch)
        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3, max_tokens=1500,
        )
        lines = [l.strip() for l in response.choices[0].message.content.split('\n') if len(l.strip()) > 10]
        all_cf.extend(lines[:batch_size])
        time.sleep(0.5)
    return all_cf

print('Generating controlled label-1 samples...')
controlled_pos = generate_controlled(PROMPT_POSITIVE, n_per_batch=100, n_batches=4)
print('Generating controlled label-0 samples...')
controlled_neg = generate_controlled(PROMPT_NEGATIVE, n_per_batch=100, n_batches=4)
print('Generating counterfactual pairs...')
cf_pos = generate_counterfactuals(controlled_pos[:60])
cf_neg = generate_counterfactuals(controlled_neg[:60])

ctrl_df = pd.DataFrame(
    [(s, 1, 'controlled') for s in controlled_pos + cf_pos] +
    [(s, 0, 'controlled') for s in controlled_neg + cf_neg],
    columns=['sentence', 'label', 'source']
)
ctrl_df.to_csv(CONTROLLED_PATH, index=False)
print(f'\nGenerated {len(ctrl_df)} controlled samples → {CONTROLLED_PATH}')
print(f'Label distribution: {ctrl_df.label.value_counts().to_dict()}')