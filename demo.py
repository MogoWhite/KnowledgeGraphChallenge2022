import pandas as pd
import re
from deep_translator import GoogleTranslator
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

csv_path = 'query/query-high.csv'
df = pd.read_csv(csv_path, encoding="utf8")

def extract_path(row):
  # Extract the path from the URL
  row['a'] = re.search(r'/([^/]*)$', row['a']).group(1)
  row['o'] = re.search(r'/([^/]*)$', row['o']).group(1)
  row['action'] = re.search(r'/([^/]*)$', row['action']).group(1)
  return row

df = df.apply(extract_path, axis=1)

print(df.loc[:, ['a', 'o','action']])

pattern1 = r'_scene1|\d+'
pattern2 = r'_'

text_list = []
for index, row in df.iterrows():
    row['o'] = re.sub(pattern1, '', row['o'])
    row['o'] = GoogleTranslator(source='auto',target='ja').translate(row['o'])

    row['a'] = re.sub(pattern1, '', row['a'])
    row['a'] = re.sub(pattern2, ' ', row['a'])
    row['a'] = GoogleTranslator(source='auto',target='ja').translate(row['a'])
    row['action'] = GoogleTranslator(source='auto',target='ja').translate(row['action'])

    text = f"高齢者が屋内で{row['a']}ことをするために{row['o']}を{row['action']}ことのリスクは何ですか？"
    text_list.append(text)

for text in text_list:
  print(text)

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")

if torch.cuda.is_available():
    model = model.to("cuda")

for prompt in text_list:
    num = 1

    input_ids = tokenizer.encode(prompt, return_tensors="pt",add_special_tokens=False).to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100, # 最長の文章長
            min_length=100, # 最短の文章長
            do_sample=True,
            top_k=500, # 上位{top_k}個の文章を保持
            top_p=0.95, # 上位{top_p}%の単語から選択する。例）上位95%の単語から選んでくる
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # bad_word_ids=[[tokenizer.unk_token_id]],
            num_return_sequences=num # 生成する文章の数
        )
    decoded = tokenizer.batch_decode(output,skip_special_tokens=True)
    for i in range(num):
        print(decoded[i])