# src/config.py
# Change DATASET to switch. Options: 'depression', 'sst2', 'hate'

DATASET = 'depression'  # <-- change this only

CONFIGS = {
    'sst2': {
        'train_path': 'data/raw/sst2_train.csv',
        'val_path': 'data/raw/sst2_val.csv',
        'domain': 'movie review sentiment',
        'label_1_name': 'positive movie review expressing enjoyment or praise',
        'label_0_name': 'negative movie review expressing disappointment or criticism',
        'generate_prompt_positive': 'positive movie review where someone enjoyed the film',
        'generate_prompt_negative': 'negative movie review where someone was disappointed with the film',
        'results_dir': 'results/sst2',
        'embeddings_dir': 'data/embeddings/sst2',
        'synthetic_suffix': 'sst2',
    },
    'depression': {
        'train_path': 'data/raw/depression_train.csv',
        'val_path': 'data/raw/depression_val.csv',
        'domain': 'mental health Reddit post',
        'label_1_name': 'person expressing depression, hopelessness, sadness, or emotional distress',
        'label_0_name': 'person expressing normal thoughts, daily activities, or positive emotions',
        'generate_prompt_positive': 'Reddit post from someone expressing depression, sadness, hopelessness, or emotional exhaustion',
        'generate_prompt_negative': 'Reddit post from someone talking about normal daily life, hobbies, positive experiences, or neutral topics',
        'results_dir': 'results/depression',
        'embeddings_dir': 'data/embeddings/depression',
        'synthetic_suffix': 'depression',
    },
    'hate': {
        'train_path': 'data/raw/hate_train.csv',
        'val_path': 'data/raw/hate_val.csv',
        'domain': 'social media post',
        'label_1_name': 'hateful or offensive tweet targeting a person or group',
        'label_0_name': 'normal or non-offensive tweet',
        'generate_prompt_positive': 'offensive or hateful social media post targeting people based on identity',
        'generate_prompt_negative': 'normal non-offensive social media post about daily life or opinions',
        'results_dir': 'results/hate',
        'embeddings_dir': 'data/embeddings/hate',
        'synthetic_suffix': 'hate',
    },
}

cfg = CONFIGS[DATASET]
TRAIN_PATH = cfg['train_path']
VAL_PATH = cfg['val_path']
DOMAIN = cfg['domain']
LABEL_1_NAME = cfg['label_1_name']
LABEL_0_NAME = cfg['label_0_name']
PROMPT_POSITIVE = cfg['generate_prompt_positive']
PROMPT_NEGATIVE = cfg['generate_prompt_negative']
RESULTS_DIR = cfg['results_dir']
EMBEDDINGS_DIR = cfg['embeddings_dir']
SUFFIX = cfg['synthetic_suffix']

# Derived paths — everything uses these
RAW_SYNTHETIC_PATH = f'data/synthetic/{SUFFIX}_raw_augmented.csv'
CONTROLLED_PATH = f'data/synthetic/{SUFFIX}_controlled.csv'
COMBINED_C2_PATH = f'data/synthetic/{SUFFIX}_combined_c2.csv'
FILTERED_PATH = f'data/filtered/{SUFFIX}_basap_filtered.csv'
MISMATCHES_PATH = f'{RESULTS_DIR}/label_mismatches.json'
DIAGNOSTICS_PATH = f'{RESULTS_DIR}/diagnostics_report.json'