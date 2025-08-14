import re
import string
import nltk
from textblob import TextBlob
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

nltk.download('punkt_tab', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')
nltk.download('averaged_perceptron_tagger_eng', download_dir='nltk_data')
nltk.data.path.append('nltk_data')

SELECTED_EDA = [
    'special_chars_count', 'n_unique_words', 'num_chars',
    'punctuation_count', 'noun_ratio', 'subjectivity',
    'stopword_ratio', 'num_words', 'adj_ratio', 'avg_sentence_length'
]

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNCTUATION = set(string.punctuation)

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

scaler_eda = joblib.load('models/scaler_eda.pkl')
scaler_bert = joblib.load('models/bert_scaler.joblib')
pca_bert = joblib.load('models/bert_pca.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
tfidf_svd = joblib.load('models/tfidf_svd_transformer.joblib')
best_model = joblib.load('models/catboost_model.joblib')

def extract_eda_features(text, selected=SELECTED_EDA):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    total_tokens = len(words) or 1
    feats = {}
    if 'num_chars' in selected:
        feats['num_chars'] = len(text)
    if 'special_chars_count' in selected:
        feats['special_chars_count'] = len(re.findall(r'[^A-Za-z0-9\s]', text))
    if 'punctuation_count' in selected:
        feats['punctuation_count'] = sum(1 for ch in text if ch in PUNCTUATION)
    if 'num_words' in selected:
        feats['num_words'] = len(words)
    if 'n_unique_words' in selected:
        feats['n_unique_words'] = len(set(w.lower() for w in words if w.isalpha()))
    if 'stopword_ratio' in selected:
        feats['stopword_ratio'] = sum(1 for w in words if w.lower() in STOPWORDS) / total_tokens
    if 'avg_sentence_length' in selected:
        feats['avg_sentence_length'] = sum(len(nltk.word_tokenize(s)) for s in sentences) / (len(sentences) or 1)
    if 'subjectivity' in selected:
        feats['subjectivity'] = TextBlob(text).sentiment.subjectivity
    if 'noun_ratio' in selected or 'adj_ratio' in selected:
        pos_tags = nltk.pos_tag(words)
        counts = {}
        for _, tag in pos_tags:
            counts[tag] = counts.get(tag, 0) + 1
        if 'noun_ratio' in selected:
            noun_count = sum(counts.get(x,0) for x in ['NN','NNS','NNP','NNPS'])
            feats['noun_ratio'] = noun_count / total_tokens
        if 'adj_ratio' in selected:
            adj_count = sum(counts.get(x,0) for x in ['JJ','JJR','JJS'])
            feats['adj_ratio'] = adj_count / total_tokens
    return [feats[f] for f in selected]


def extract_mean_pooling_vector(text, tokenizer, model, max_len=512, stride=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoded = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=max_len, stride=stride,
                        return_overflowing_tokens=True, padding='max_length')
    ids_chunks = encoded['input_ids']
    mask_chunks = encoded['attention_mask']
    vecs = []
    model.to(device).eval()
    with torch.no_grad():
        for ids, mask in zip(ids_chunks, mask_chunks):
            ids = ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            out = model(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).expand(out.size())
            summed = (out * m).sum(dim=1)
            count = m.sum(dim=1)
            vecs.append((summed/count).squeeze(0))
    return torch.stack(vecs).mean(dim=0).cpu().numpy()


def extract_bert_features(text1, text2):
    v1 = extract_mean_pooling_vector(text1, bert_tokenizer, bert_model)
    v2 = extract_mean_pooling_vector(text2, bert_tokenizer, bert_model)
    return np.concatenate([v1, v2, v1-v2, v1*v2])

def get_feature_vector(t1, t2):
        # EDA features (t1 - t2)
        diff1 = extract_eda_features(t1, SELECTED_EDA)
        diff2 = extract_eda_features(t2, SELECTED_EDA)
        eda_diff = scaler_eda.transform([np.subtract(diff1, diff2)])[0]

        # BERT features for (t1, t2)
        bert_feat = extract_bert_features(t1, t2)
        bert_s = scaler_bert.transform([bert_feat])[0]
        bert_p = pca_bert.transform([bert_s])[0]

        # TF-IDF for "t1 [SEP] t2"
        pair = f"{t1} [SEP] {t2}"
        tf = tfidf_vectorizer.transform([pair]).toarray()
        tf_p = tfidf_svd.transform(tf)[0]

        return np.hstack([bert_p, tf_p, eda_diff])


def predict_real_text_id(text1, text2):
        # Predict with original order: (text1, text2)
        feat_orig = get_feature_vector(text1, text2)
        prob_orig = best_model.predict_proba([feat_orig])[0][1]

        # Predict with swapped order: (text2, text1)
        feat_swapped = get_feature_vector(text2, text1)
        prob_swapped = best_model.predict_proba([feat_swapped])[0][1]

        prob_text1_real = (prob_orig + (1 - prob_swapped)) / 2

        if prob_text1_real >= 0.5:
            return 1
        else:
            return 2



