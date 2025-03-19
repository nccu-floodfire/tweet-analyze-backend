import bitermplus as btm
import numpy as np
import pandas as pd
import re
import emoji
import jieba
import tmplot as tmp

jieba.load_userdict('./dictionary.txt')

stop_words = set()
with open('./cn_stopwords.txt', 'r', encoding='utf-8') as file:
    for line in file:
        stop_words.add(line.strip())

def clean_text(text):
        text = emoji.demojize(text)
        text = re.sub(r'RT @\w+:', '', text)  
        text = re.sub(r'https?://\S+', '', text)  
        text = re.sub(r'@\w+', '', text)  
        return text

def clean_text_chinese_with_stopwords(text):
        words = jieba.cut(text)
        filtered_words = [word for word in words if word not in stop_words]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

def evaluate_topic_models(data, T_values, M=20, alpha_scale=50, beta=0.01, iterations=25, seed=12321):
    results = []
    for T in T_values:
        # print(f"Evaluating T={T}")
        model = btm.BTM(
            data['X'], data['vocabulary'], seed=seed, T=T, M=M, alpha=alpha_scale/T, beta=beta
        )
        model.fit(data['biterms'], iterations=iterations)
        p_zd = model.transform(data['docs_vec'])

        perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, data['X'], T)
        coherence = btm.coherence(model.matrix_topics_words_, data['X'], M=M)
        
        results.append({
            'T': T,
            'Perplexity': perplexity,
            'Coherence': coherence
        })

    return pd.DataFrame(results)

def preprocess_data(texts):
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    biterms = btm.get_biterms(docs_vec)
    return {
        'X': X,
        'vocabulary': vocabulary,
        'vocab_dict': vocab_dict,
        'docs_vec': docs_vec,
        'biterms': biterms
    }



def btm_analysis(data, path=None):
    #read txt file
    if path:
        jieba.load_userdict(path)
    
    data = data[data['lang']=='zh']
    selec = ['created_at','from_user_name','from_user_realname','text']
    data = data[selec]
    data['clean_text'] = data['text'].apply(clean_text)
    data = data.drop_duplicates(subset='clean_text', keep='first')
    data['clean_text'] = data['clean_text'].apply(clean_text_chinese_with_stopwords)
    data['prefix'] = data['clean_text'].apply(lambda x: x[:10])
    data = data.drop_duplicates(subset='prefix', keep='first')
    data = data.drop(columns=['prefix'])
    origin = data["text"].tolist()
    text = data["clean_text"].tolist()
    
    # decide the number of topics
    data1 = preprocess_data(text)
    T_values = range(1, 10) 
    evaluation_results = evaluate_topic_models(data1, T_values)
    evaluation_results['First_Coherence'] = evaluation_results['Coherence'].apply(lambda x: x[0])
    best_model_idx = evaluation_results['First_Coherence'].idxmax()  # 或者使用 'Coherence'.idxmax() 'Perplexity'].idxmin()
    T = evaluation_results.loc[best_model_idx, 'T']

    # BTM
    X, vocabulary, vocab_dict = btm.get_words_freqs(text)
    tf = np.array(X.sum(axis=0)).ravel()
    docs_vec = btm.get_vectorized_docs(text, vocabulary)
    docs_lens = list(map(len, docs_vec))

    biterms = btm.get_biterms(docs_vec)

    model = btm.BTM(
        X, vocabulary, seed=12321, T=T, M=20, alpha=50/8, beta=0.01)
    model.fit(biterms, iterations=25)
    p_zd = model.transform(docs_vec)

    model.labels_

    #distribution of topics
    topics_coords = tmp.prepare_coords(model)

    #terms probability for each topic
    phi = tmp.get_phi(model)
    terms_probs = {}
    for i in range(T):
        terms_prob = tmp.calc_terms_probs_ratio(phi, topic=i, lambda_=1)
        terms_probs[i] = terms_prob

    #top 5 doc for each topic
    data = data.reset_index()
    # data.columns = ['index','created_at','from_user_name','from_user_realname','text','clean_text','Topic0','Topic1','Topic2','Topic3','Topic4']
    theta = tmp.get_theta(model)
    theta = theta.transpose()
    data = pd.concat([data,theta], axis = 1)
    top_5_doc = {}
    for i in range(T):
        top_5_rows = data.nlargest(5, {i})
        top_5_doc[i] = top_5_rows

    return topics_coords, terms_probs, top_5_doc
