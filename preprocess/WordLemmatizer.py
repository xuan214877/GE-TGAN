import nltk
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import os

nltk.data.path.append(r'E:/preprocess/nltk_data')

current_dir = os.getcwd()
current_dir1 =r"E:/preprocess/data"
def lemmatize_text(text):
    def get_wordnet_pos(tag):
        tag = tag[0].upper() 
        if tag in ['J']:
            return wordnet.ADJ
        elif tag in ['V', 'D']:
            return wordnet.VERB
        elif tag in ['N']:
            return wordnet.NOUN
        elif tag in ['R']:
            return wordnet.ADV
        else:
            return None 

    if not isinstance(text, str) or pd.isnull(text):
        return None
    sentence = text.lower()
    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for word, tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos is not None: 
            lemmas_sent.append(wnl.lemmatize(word, pos=wordnet_pos))
        else:
            lemmas_sent.append(word)
    return " ".join(lemmas_sent)

for filename in os.listdir(current_dir1):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        input_file_path = os.path.join(current_dir1, filename)
        try:
            df = pd.read_excel(input_file_path)
            df1 = df[['Author', 'Title', 'Abstract', 'Author keywords', 'Index keywords','Merged']].copy()

            endcont = []
            for i in tqdm(df1['Merged'].values):
                lemmatized_text = lemmatize_text(i)
                endcont.append(lemmatized_text)

            df1['WordNetLemmatizer'] = endcont

            base_name = os.path.splitext(filename)[0]
            output_excel_path = os.path.join(current_dir, f'{base_name}_lemmatized.xlsx')
            output_json_path = os.path.join(current_dir, f'{base_name}_lemmatized.json')

            df1.to_excel(output_excel_path)
            df1.to_json(output_json_path)
        except pd.errors.ParserError as pe:
            print(f"{filename}: {pe}")
        except Exception as e:
            print(f"{filename}: {e}")