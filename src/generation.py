# src/generation.py
import os, time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from config import PROMPT_POSITIVE, PROMPT_NEGATIVE, DOMAIN, RAW_SYNTHETIC_PATH

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])

os.makedirs('data/synthetic', exist_ok=True)

def generate_unconstrained(condition_description, n_samples=200):
    all_lines = []
    for batch in range(2):
        prompt = f'''Generate 100 short {DOMAIN} sentences.
Description: {condition_description}
Format: one sentence per line, no numbering, no quotes, no extra text.
Write naturally as a real person would.
Return only the sentences.'''
        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.8, max_tokens=2000,
        )
        lines = [l.strip() for l in response.choices[0].message.content.split('\n') if len(l.strip()) > 10]
        all_lines.extend(lines[:100])
        time.sleep(0.5)
    return all_lines[:n_samples]

print(f'Generating label-1 samples...')
positive_samples = generate_unconstrained(PROMPT_POSITIVE, 200)
print(f'Generating label-0 samples...')
negative_samples = generate_unconstrained(PROMPT_NEGATIVE, 200)

synth_df = pd.DataFrame({
    'sentence': positive_samples + negative_samples,
    'label': [1]*len(positive_samples) + [0]*len(negative_samples),
    'source': 'synthetic_raw'
})

synth_df.to_csv(RAW_SYNTHETIC_PATH, index=False)
print(f'Generated {len(synth_df)} samples → {RAW_SYNTHETIC_PATH}')
print(f'Label distribution: {synth_df.label.value_counts().to_dict()}')