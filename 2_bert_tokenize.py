# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import transformers as ppb

# -------------------------------------------------------
# CONFIG — change name to 'sample_100' or 'sample_20' etc.
# -------------------------------------------------------
name = 'sample_100'

'''Import Data'''
df = pd.read_csv(name + '.csv')

# Drop rows where notesCleaned is missing/NaN
df = df.dropna(subset=['notesCleaned']).reset_index(drop=True)

# Ensure text column is string type
df['notesCleaned'] = df['notesCleaned'].astype(str)

df.loc[df['class'] == 'suicide', 'classification_int'] = 1
df.loc[df['class'] == 'non-suicide', 'classification_int'] = 0

notesCleaned = df['notesCleaned']

'''Load pretrained BERT tokenizer'''
model_class, tokenizer_class, pretrained_weights = (
    ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased'
)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

'''Tokenize — truncate to 512 tokens to avoid BERT limit errors'''
tokenized = notesCleaned.apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True)
)

tokens = pd.Series([], dtype=str)
max_value = 0

print(f"\nTokenizing {name}: {df.shape[0]} rows")
for i in range(len(df)):
    tokens[i] = tokenized[i]
    if len(tokenized[i]) >= max_value:
        max_value = len(tokenized[i])

'''Export'''
df['max_count'] = max_value
df['tokens'] = tokens
df.to_csv(name + '_tokens.csv', index=False)
print(f"Done! Saved to {name}_tokens.csv (max token length: {max_value})")
