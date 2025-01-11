import streamlit as st
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.stops.words import Stops
from cltk.wordnet.wordnet import WordNetCorpusReader
from cltk.core.exceptions import CLTKException
from cltk.data.fetch import FetchCorpus
from cltk import NLP
import stanza
import string
import re
import os
from collections import Counter

st.write("Working Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

# Programmatically approve CLTK downloads
os.environ["CLTK_DATA_DOWNLOAD"] = "true"  # Bypass download confirmation prompts

@st.cache_resource
def initialize_stanza():
    stanza_dir = 'stanza_resources/la'  # Relative path
    if not os.path.exists(stanza_dir):
        st.write("Downloading the model, please wait...")
        stanza.download('la')
    nlp = stanza.Pipeline(lang='la', package='proiel', processors='tokenize,mwt,pos', verbose=False)
    return nlp
# Use the function to initialize the model
nlp = initialize_stanza()

@st.cache_resource
def ensure_cltk_models():
    """Ensure the required CLTK models are downloaded."""
    cltk_data_path = os.path.expanduser("~/cltk_data")  # Default CLTK data path
    if not os.path.exists(os.path.join(cltk_data_path, "lat/model/lat_models_cltk")):
        st.write("Downloading required CLTK models, please wait...")
        fetch_corpus = FetchCorpus(language="lat")
        fetch_corpus.import_corpus("lat_models_cltk")
        st.write("CLTK models downloaded successfully.")

ensure_cltk_models()

@st.cache_resource
def initialize_cltk():
    """Initialize the CLTK pipeline for Latin."""
    cltk_nlp = NLP(language="lat")
    return cltk_nlp
# Use the function to initialize CLTK
cltk_nlp = initialize_cltk()


def rr_75(sentence_2):
    # We find here the implementation of the above defined function 'rr_75'
    lemmatizer = LatinBackoffLemmatizer()
    with open('basic_voces.txt', 'r') as f:
        corpus = f.read()
    corpus = re.sub(r'\d+', '', corpus)
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    words_corpus = corpus.lower().split()
    lemmas_corpus = [re.sub(r'[1-4]$', '', lemma[1]) for lemma in lemmatizer.lemmatize(words_corpus)]
    lemma_counts = Counter(lemmas_corpus)
    ranked_lemmas = sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True)
    rank_count = 0
    prev_count = -1
    ranked_lemmata = []
    for rank, (lemma, count) in enumerate(ranked_lemmas):
        if count != prev_count:
            rank_count += 1
        ranked_lemmata.append((rank_count, lemma, count))
        prev_count = count
    sentence = sentence_2.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    lemmas = lemmatizer.lemmatize(words)
    lemmata = [re.sub(r'[1-4]$', '', lemma[1]) for lemma in lemmas]
    rank_sum_750 = 0
    for lemma in lemmata:
        for freq in ranked_lemmata:
            if lemma == freq[1]:
                if freq[0] >= 750:
                    rank_sum_750 += 1
    ranked_text_750 = round(rank_sum_750 / len(words), 3)
    return ranked_text_750

def whsw_ran(sentence_2):
    with open('basic_voces.txt', 'r') as f:
        corpus = f.read()
    corpus = re.sub(r'\d+', '', corpus)
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    words_corpus = corpus.lower().split()
    word_counts = Counter(words_corpus)
    ranked_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    rank_count = 0
    prev_count = -1
    rank_words = []
    for rank, (word, count) in enumerate(ranked_words):
        if count != prev_count:
            rank_count += 1
        rank_words.append((rank_count, word, count))
        prev_count = count
    sentence = sentence_2.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    lemmatizer = LatinBackoffLemmatizer()
    lemmas = lemmatizer.lemmatize(words)
    token_lemma_pairs = [(word, lemma[1].rstrip('1234')) for word, lemma in zip(words, lemmas)]
    stops_obj = Stops(iso_code="lat")
    filtered_lemmas = stops_obj.remove_stopwords(tokens=[lemma[1].rstrip('1234') for lemma in lemmas])
    tokens_filtered = [pair[0] for pair in token_lemma_pairs if pair[1] in filtered_lemmas]
    rank_sum = 0
    for word in tokens_filtered:
        for freq in rank_words:
            if word == freq[1]:
                rank_sum += freq[0]
    ranked_text = round(rank_sum / len(tokens_filtered), 3)
    return ranked_text

def verb_num(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.lower()
    words = sentence.split()
    doc = nlp(sentence)
    annotations = []
    person_3sg = []
    person_2pl = []
    for sent in doc.sentences:
        for word in sent.words:
            feats = word.feats
            if feats is not None and 'VerbForm=Fin' in feats:
                annotations.append(feats)
                if 'Person' in feats and 'Number' in feats:
                    feats_dict = {key: value for key, value in [f.split('=') for f in feats.split('|')]}
                    if feats_dict['Person'] == '3' and feats_dict['Number'] == 'Sing':
                        person_3sg.append(word.text)
                    elif feats_dict['Person'] == '2' and feats_dict['Number'] == 'Plur':
                        person_2pl.append(word.text)
    verbcount = len(annotations)
    person_3sg_count = len(person_3sg)
    person_2pl_count = len(person_2pl)
    ratio_3sg = round(person_3sg_count / verbcount, 3)
    ratio_2pl = round(person_2pl_count / verbcount, 3)
    return ratio_3sg, ratio_2pl

def pqp(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    doc = nlp(sentence)
    annotations = []
    pqperf = []
    for sent in doc.sentences:
        for word in sent.words:
            feats = word.feats
            if feats is not None and 'VerbForm=Fin' in feats:
                annotations.append(feats)
                if 'Tense' in feats:
                    feats_dict = {key: value for key, value in [f.split('=') for f in feats.split('|')]}
                    if feats_dict['Tense'] == 'Pqp':
                        pqperf.append(word.text)
    verbcount = len(annotations)
    pqperf_count = len(pqperf)
    ratio_pqperf = round(pqperf_count / verbcount, 3)
    return ratio_pqperf

def synsw(sentence_2):
    LWN = WordNetCorpusReader(iso_code="lat")
    sentence = sentence_2.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    lemmatizer = LatinBackoffLemmatizer()
    lemmas = lemmatizer.lemmatize(words)
    lemmata = [re.sub(r'[1-4]$', '', lemma[1]) for lemma in lemmas]
    stops_obj = Stops(iso_code="lat")
    tokens_filtered = stops_obj.remove_stopwords(tokens=lemmata)
    total_synonyms = 0
    for lemma in tokens_filtered:
        if lemma:
            try:
                lemma_synsets = LWN.lemma(lemma)
                if lemma_synsets:
                    synsets = list(lemma_synsets[0].synsets())
                    num_synonyms = len(synsets)
                    total_synonyms += num_synonyms
                else:
                    num_synonyms = 0
            except Exception:
                continue
    avg_synonyms = round(total_synonyms / len(words), 3) if len(words) > 0 else 0
    return avg_synonyms

def ttrsw(sentence_2):
    sentence = sentence_2.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    lemmatizer = LatinBackoffLemmatizer()
    lemmas = lemmatizer.lemmatize(words)
    lemmata = [re.sub(r'[1-4]$', '', lemma[1]) for lemma in lemmas]
    stops_obj = Stops(iso_code="lat")
    tokens_filtered = stops_obj.remove_stopwords(tokens=lemmata)
    unique_lemmas = set(tokens_filtered)
    total_lemmas = len(tokens_filtered)
    original_ttr = round(len(unique_lemmas) / total_lemmas, 3)
    return original_ttr

def formula(verschr_RS, R_K, R_SA, ger_pred, Gerundium, Komps, Abl_qual, Dat_auct, Satzt, TU, ranked_text_750, ranked_text, ratio_3sg, ratio_2pl, ratio_pqperf, avg_synonyms, original_ttr, sentence):
    words = sentence.split()
    print("formula is intinialized") #debug
    SE_optimum = (3.78*verschr_RS + 2.73*R_K + 2.39*R_SA + 2.29*ger_pred + 2.76*Gerundium + 2.79*Komps + 2.25*Abl_qual + 1.98*Dat_auct) / len(words)
    stt = Satzt / TU
    readability = (
        14.478 +
        24.885 * ranked_text_750 +
        9.872 * ratio_2pl -
        0.015 * ranked_text -
        9.473 * ratio_pqperf -
        15.215 * SE_optimum +
        0.402 * stt -
        0.097 * avg_synonyms +
        2.395 * ratio_3sg -
        7.141 * original_ttr
    )

    # Threshold checks
    if not (0 <= ranked_text_750 <= 0.35):
        st.warning("The value for 'rr_75' (percentage of words outside a list of the most common 750 words) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (0 <= ratio_2pl <= 0.16):
        st.warning("The value for '2pl' (percentage of verbs in 2nd plural) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (368.12 <= ranked_text <= 704.56):
        st.warning("The value for 'whsw_ran' (frequency of word forms sorted by rank) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (0 <= ratio_pqperf <= 0.26):
        st.warning("The value for 'pqp' (percentage of verbs in pluperfect) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (0 <= SE_optimum <= 0.18):
        st.warning("The value for 'SE_optimum' (group of optimal syntactical phenomena) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (0 <= stt <= 5.07):
        st.warning("The value for 'stt' (sentence depth with regards to number of T-units) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (1.52 <= avg_synonyms <= 25.34):
        st.warning("The value for 'synsw' (number of synsets without function words) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (0.03 <= ratio_3sg <= 1.13):
        st.warning("The value for '3sg' (percentag of verbs in 3rd singular) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")
    if not (0.7 <= original_ttr <= 1.03):
        st.warning("The value for 'ttrsw' (TTR without function words) exceeds the standard deviation more than threefold. Be careful when interpreting the readability score.")

    # print for internal checks
    print(f"ranked_text_750: {ranked_text_750}")
    print(f"ratio_2pl: {ratio_2pl}")
    print(f"ranked_text: {ranked_text}")
    print(f"ratio_pqperf: {ratio_pqperf}")
    print(f"SE_optimum: {SE_optimum}")
    print(f"stt: {stt}")
    print(f"avg_synonyms: {avg_synonyms}")
    print(f"ratio_3sg: {ratio_3sg}")
    print(f"original_ttr: {original_ttr}")


    return round(readability, 2)


# Streamlit UI components
st.title("Latin Text Readability Calculator")

# User input
sentence = st.text_area("Copy here the complete Latin text:")
sentence_2 = st.text_area("Copy here the complete Latin text without the given annotations:")
verschr_RS = st.number_input("Enter number of instances of 'verschr_RS':", min_value=0.0, help="Count the number of instances of 'verschränkter Relativsatz'.")
R_K = st.number_input("Enter number of instances of 'Relativsatz mit Konjunktiv':", min_value=0.0, help="Count the number of instances of 'Relativsatz mit Konjunktiv'.")
R_SA = st.number_input("Enter number of instances of 'Relativischer Satzanschluss':", min_value=0.0, help="Count the number of instances of 'Relativischer Satzanschluss'.")
ger_pred = st.number_input("Enter number of instances of 'prädikatives Gerundivum':", min_value=0.0, help="Count the number of instances of 'Gerundivum mit esse'.")
Gerundium = st.number_input("Enter number of instances of 'Gerundium':", min_value=0.0, help="Count the number of instances of 'Gerundium'.")
Komps = st.number_input("Enter number of instances of 'Komparativsatz':", min_value=0.0, help="Count the number of instances of 'Komparativsatz'.")
Abl_qual = st.number_input("Enter number of instances of 'Ablativus qualitatis':", min_value=0.0, help="Count the number of instances of 'Ablativus Qualitatis'.")
Dat_auct = st.number_input("Enter number of instances of 'Dativus auctoris':", min_value=0.0, help="Count the number of instances of 'Dativus Auctoris'.")
Satzt = st.number_input("Enter value for 'Satztiefe':", min_value=0.0, help="To calculate the sentence depth each predicate gets a number: A predicate of a matrix sentence gets number '0', a predicate of a clause dependent on the matrix sentence gets number '1'; a clause dependent of subclause whose predicate has number '1' gets number '2' and so on. Sum up the numbers of all the predicates in the given text.")
TU = st.number_input("Enter number of 'T-Units':", min_value=0.0, help="A T-Unit is a matrix sentence with all its dependent subclauses. Count the number of instances of T-Units.")

###st.sidebar.info(
   # """
   # **Important Information**

   # Currently, the numbers are not accurate due to an expired certificate of the Latin WordNet ([latinwordnet.exeter.ac.uk](https://latinwordnet.exeter.ac.uk)).
   # """
#)###

if st.button("Calculate Readability"):
   
    # Calculate variables
    ranked_text_750 = rr_75(sentence_2)
    ranked_text = whsw_ran(sentence_2)
    ratio_3sg, ratio_2pl = verb_num(sentence)
    ratio_pqperf = pqp(sentence)
    avg_synonyms = synsw(sentence_2)
    original_ttr = ttrsw(sentence_2)

    # Calculate readability score
    readability_score = formula(verschr_RS, R_K, R_SA, ger_pred, Gerundium, Komps, Abl_qual, Dat_auct, Satzt, TU, ranked_text_750, ranked_text, ratio_3sg, ratio_2pl, ratio_pqperf, avg_synonyms, original_ttr, sentence)

    # Display the output
    st.success(f"Readability Score: {readability_score}")
