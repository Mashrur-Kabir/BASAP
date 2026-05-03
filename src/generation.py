# src/generation.py
# IMPROVEMENT: Generate 200 per label (was 100) = 400 total raw synthetic samples
import os, time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])

def generate_unconstrained(label_name, n_samples=200):
    '''Generate synthetic samples with NO bias controls (Condition 2)'''
    all_lines = []
    # Split into 2 batches of 100 to stay within token limits
    for batch in range(2):
        prompt = f'''Generate 100 short movie review sentences.
Label: {label_name}
Format: one review per line, no numbering, no quotes.
Just return the sentences.'''

        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.8,
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
        all_lines.extend(lines[:100])
        time.sleep(0.5)
    return all_lines[:n_samples]

# Generate 400 total samples (200 positive, 200 negative)
print('Generating positive samples...')
positive_samples = generate_unconstrained('positive', 200)
print('Generating negative samples...')
negative_samples = generate_unconstrained('negative', 200)

synth_df = pd.DataFrame({
    'sentence': positive_samples + negative_samples,
    'label': [1]*len(positive_samples) + [0]*len(negative_samples),
    'source': 'synthetic_raw'
})

synth_df.to_csv('data/synthetic/sst2_raw_augmented.csv', index=False)
print(f'Generated {len(synth_df)} synthetic samples')
print(f'Label distribution: {synth_df.label.value_counts().to_dict()}')