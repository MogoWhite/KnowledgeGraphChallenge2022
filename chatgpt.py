import pandas as pd
import re
from deep_translator import GoogleTranslator
import openai

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

# APIキーを取得
openai.api_key = "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"

#モデルを指定
model_engine = "text-davinci-002"

#指示（Prompt）を設定
question = text_list[100] 
prompt = f'''I am a highly intelligent question answering bot. 
Q: {question}
A:'''

#推論を実行
response = openai.Completion.create(
    engine="text-davinci-003", 
    prompt=prompt, 
    max_tokens=1024,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop='/.')

result = prompt+response['choices'][0]['text']


# プロンプト
prompt_1 = result+'\n'"Q: リスクを減らす方法は？\nA:"

#危険な状況が改善の代替案
response_1= openai.Completion.create(
    engine="text-davinci-003", 
    prompt=prompt_1, 
    max_tokens=1024,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,

    stop='/.')

result = prompt_1+response_1['choices'][0]['text']
print(result)