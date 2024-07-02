import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# NLTK'nin VADER duygu analizi modülünü yükleyin
nltk.download('vader_lexicon')

# Duygu analizi yapmak için SentimentIntensityAnalyzer'ı başlatın
analyzer = SentimentIntensityAnalyzer()

# Çeviri yapmak için GoogleTranslator'ı başlatın
translator = GoogleTranslator(source='tr', target='en')

# Excel dosyasını okuyun
excel_file_path = 'metinler.xlsx'  # Excel dosya yolunuz
df = pd.read_excel(excel_file_path)

# Sütun adlarını ve ilk birkaç satırı görüntüleyin
print("Sütun Adları:", df.columns)
print("İlk birkaç satır:\n", df.head())

# Yıldız sonuçlarını depolamak için bir liste oluşturun
star_ratings_list = []

# Excel dosyasındaki her bir metni işleyin
for index, row in df.iterrows():
    turkish_text = row[df.columns[0]]  # İlk sütundaki metni alın

    # Metni İngilizceye çevirin
    translated_text = translator.translate(turkish_text)

    # Çevirilen metin üzerinde duygu analizi yapın
    scores = analyzer.polarity_scores(translated_text)

    # Negatif, nötr ve pozitif duygusal skorları alın
    neg = scores['neg']
    neu = scores['neu']
    pos = scores['pos']

    # Negatif, nötr ve pozitif skorlara göre yıldız derecelendirmesi
    if neg >= 0.5:
        star_rating = 0
    elif pos >= 0.5:
        star_rating = 5
    elif pos >= 0.3:
        star_rating = 4
    elif neu >= 0.5:
        star_rating = 3
    elif neg >= 0.3:
        star_rating = 2
    else:
        star_rating = 1

    # Yıldız derecelendirmesini listeye ekleyin
    star_ratings_list.append(star_rating)

    # Yıldızları görüntüle
    stars = '*' * star_rating + '-' * (5 - star_rating)

    # Sonuçları görüntüleyin
    print(f"Satır {index + 1} - Türkçe Metin: {turkish_text}")
    print(f"İngilizce Metin: {translated_text}")
    print(f"Duygusal Skorlar: {scores}")
    print(f"Yıldız Derecelendirmesi: {stars}\n")

# Toplu yıldız derecelendirmeleri göster
print("Tüm Yorumların Yıldız Derecelendirmeleri:")
for idx, rating in enumerate(star_ratings_list, start=1):
    stars = '*' * rating + '-' * (5 - rating)
    print(f"Yorum {idx}: {stars}")
    
    # Ortalama yıldız derecelendirmesi hesapla
average_rating = sum(star_ratings_list) / len(star_ratings_list)
average_stars = '*' * round(average_rating) + '-' * (5 - round(average_rating))

# Genel ortalama yıldız derecesini göster
print(f"Genel Ortalama: {average_stars}")