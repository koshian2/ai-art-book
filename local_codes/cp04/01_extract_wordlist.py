import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

def extract_wordlist():
    freq_words = []
    # 頻出語2万個登録
    with open("data/enwiki-2023-04-13.txt", "r", encoding="utf-8") as fp:
        data = fp.read()
        for i, f in enumerate(data.split("\n")):
            sep = f.split(" ")
            freq_words.append(sep[0])
            if i >= 20000-1:
                break
    # WordNetから列挙して、頻出語に含まれていれば登録
    wordsets = set()
    for i, s in enumerate(wn.all_synsets(wn.NOUN)):
        for w in s.lemmas():
            wstr = str(w.name())
            # 3文字以上の名詞
            if wstr in freq_words and len(wstr) >= 3:
                wordsets.add(wstr)
    with open("data/corpus_20k.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(wordsets))

if __name__ == "__main__":
    extract_wordlist()