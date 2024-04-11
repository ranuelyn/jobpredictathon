# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:02:10 2024

@author: yusuf
"""
# the libraries


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup # for data cleaning
import category_encoders as ce # for binary encoding
import re
from sklearn.feature_extraction.text import CountVectorizer

#%%

dataAday = pd.read_csv('data_aday_log.csv')
dataCv = pd.read_csv('data_cv_details.csv')
dataJob = pd.read_csv('data_job_details.csv')
dataTest = pd.read_csv('son2_basvurular_test.csv')

#%% What is in it?

# Her bir veri seti için yapısal özellikleri kontrol edin
print("Aday Logları Yapısal Özellikleri:")
print(dataAday.info())
print("\nCV Detayları Yapısal Özellikleri:")
print(dataCv.info())
print("\nİş Detayları Yapısal Özellikleri:")
print(dataJob.info())
print("\nTest Başvuruları Yapısal Özellikleri:")
print(dataTest.info())

#%% Clean the data.

def clean_html(raw_html):
    clean_text = BeautifulSoup(raw_html, "html.parser").text
    return clean_text

dataJob['jobDescription'] = dataJob['jobDescription'].apply(clean_html)

#%% Missing values.

print("Eksik veri sayısı:")
print(dataAday.isnull().sum())
print(dataCv.isnull().sum())
print(dataJob.isnull().sum())
print(dataTest.isnull().sum())

#%% Drop the NA values.

dataAday = dataAday.dropna()
dataCv= dataCv.dropna()
dataJob = dataJob.dropna()

#%% Drop "Unnamed" Column

dataAday.drop('Unnamed: 0', axis=1, inplace=True)
dataCv.drop('Unnamed: 0', axis=1, inplace=True)
dataJob.drop('Unnamed: 0', axis=1, inplace=True)

# applicationDate sütununu datetime'a dönüştürme
dataAday['applicationDate'] = pd.to_datetime(dataAday['applicationDate'])

#%% Data Types

# Veri tiplerini inceleme
print("Aday Logları Veri Tipleri:")
print(dataAday.dtypes)
print("\nCV Detayları Veri Tipleri:")
print(dataCv.dtypes)
print("\nİş Detayları Veri Tipleri:")
print(dataJob.dtypes)

# Kategorik sütunların benzersiz değerlerini inceleme
print("CV Detayları - İş Arayan Şehir Benzersiz Değer Sayısı:")
print(dataCv['jobseekerCity'].value_counts())

#dataCv_encoded = pd.get_dummies(dataCv, columns=['jobseekerCity'])


#%% 

# Tarih özelliklerini çıkarma
dataAday['applicationYear'] = dataAday['applicationDate'].dt.year
dataAday['applicationMonth'] = dataAday['applicationDate'].dt.month
dataAday['applicationDay'] = dataAday['applicationDate'].dt.day

# Zaman dilimleri oluşturma
dataAday['applicationHour'] = dataAday['applicationDate'].dt.hour
#dataAday = dataAday.drop('applicationDate', axis = 1)

#%%
# İlk olarak, her şehri ayrı bir liste elemanı olarak ayırma
expanded_jobCity = dataJob['jobCity'].str.split(',', expand=True)

# Şimdi, bu genişletilmiş DataFrame'i orijinal DataFrame'e satır olarak ekleyin
# Bu işlem, her bir şehri ayrı bir satır olarak ele alacak
expanded_jobCity = expanded_jobCity.stack().reset_index(level=1, drop=True).rename('jobCity')

# Orijinal DataFrame'e bu yeni şehir sütunu ekleyin
dataJob_expanded = dataJob.drop('jobCity', axis=1).join(expanded_jobCity)

# Sonuçları kontrol edin
print(dataJob_expanded.head())


#%% Binary Encoding for jobPosition and jobCity

# jobseekerCity için BinaryEncoder oluşturma ve uygulama
encoder = ce.BinaryEncoder()
dataCv_encoded = encoder.fit_transform(dataCv[['jobseekerCity']])

# Encoder'ın sonuçlarını göster
print(dataCv_encoded.head())
dataJob_expanded.rename(columns={'jobCity': 'jobseekerCity'}, inplace=True)
# Not: Encoder, dataCv üzerinde eğitilmiştir ve bu eğitim, aynı şehirler için dataJob'da da kullanılabilir.
dataJob_encoded = encoder.transform(dataJob_expanded[['jobseekerCity']])

# Sonuçları kontrol etme
print(dataJob_encoded.head())

dataJob_encoded.rename(columns={'jobseekerCity': 'jobCity'}, inplace=True)

#%% add the columns

# Orijinal jobseekerCity sütununu kaldırma
dataCv.drop('jobseekerCity', axis=1, inplace=True)

# İkili kodlanmış sütunları dataCv veri setine ekleyin
dataCv = pd.concat([dataCv, dataCv_encoded], axis=1)

# Orijinal jobCity sütununu kaldırma
dataJob_expanded.drop('jobCity', axis=1, inplace=True)

# İkili kodlanmış sütunları dataJob veri setine ekleyin
dataJob_expanded = pd.concat([dataJob_expanded, dataJob_encoded], axis=1)


farkli_deger_sayisi = dataJob_expanded['jobPosition'].nunique()

# Farklı değerlerin sayısını yazdırma
print(farkli_deger_sayisi)

farkli_deger_sayisi2 = dataCv['jobseekerLastPosition'].nunique()

# Farklı değerlerin sayısını yazdırma
print(farkli_deger_sayisi2)

#%% 

# jobseekerCity için BinaryEncoder oluşturma ve uygulama
encoder = ce.BinaryEncoder()
dataCv_encoded = encoder.fit_transform(dataCv[['jobseekerLastPosition']])

# Encoder'ın sonuçlarını göster
print(dataCv_encoded.head())

dataJob_expanded.rename(columns={'jobPosition': 'jobseekerLastPosition'}, inplace=True)

dataJob_encoded = encoder.transform(dataJob_expanded[['jobseekerLastPosition']])

# Sonuçları kontrol etme
print(dataJob_encoded.head())

#%%

# Orijinal jobseekerCity sütununu kaldırma
dataCv.drop('jobseekerLastPosition', axis=1, inplace=True)
# İkili kodlanmış sütunları dataCv veri setine ekleyin
dataCv = pd.concat([dataCv, dataCv_encoded], axis=1)

# Orijinal jobCity sütununu kaldırma
dataJob_expanded.drop('jobseekerLastPosition', axis=1, inplace=True)

# İkili kodlanmış sütunları dataJob veri setine ekleyin
dataJob_expanded = pd.concat([dataJob_expanded, dataJob_encoded], axis=1)

#%% last jobDescripton and departmentName!

import re
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

# Metin temizleme fonksiyonu
def clean_text(text):
    # BeautifulSoup uyarısını önlemek için kontrol
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "lxml").text  # HTML etiketlerini kaldırma
    text = text.lower()  # Küçük harfe çevirme
    text = re.sub(r'\s+', ' ', text)  # Boşlukları düzeltme
    text = re.sub(r"[^\w\s]", '', text)  # Noktalama işaretlerini kaldırma
    return text

# jobDescription sütununu temizleme
dataJob_expanded['jobDescription_clean'] = dataJob_expanded['jobDescription'].apply(clean_text)
dataCv['first_word_department2'] = dataCv['first_word_department'].apply(clean_text)


# Anahtar kelime çıkarımı
vectorizer = CountVectorizer(max_features=100)  # İlk 100 anahtar kelimeyi almak için
X = vectorizer.fit_transform(dataJob['jobDescription_clean'])

# sklearn'ün eski sürümleri için get_feature_names_out yerine get_feature_names kullanılabilir
try:
    keywords = vectorizer.get_feature_names_out()
except AttributeError:
    keywords = vectorizer.get_feature_names()

# Anahtar kelimeleri göster
print(keywords)

#%%
# İlk kelimeyi almak için fonksiyon
def get_first_word(department):
    return department.split()[0] if isinstance(department, str) else ""

dataCv['first_word_department'] = dataCv['departmentName'].apply(get_first_word)

#%%

# dataCv içindeki tecrübeyi model özelliği olarak kullanma
dataCv['totalExperienceYear_feature'] = dataCv['totalExperienceYear']

# dataJob içindeki tecrübe gereksinimlerini model özellikleri olarak kullanma
# Bu adımlar, dataJob veri seti üzerinde direkt olarak uygulanabilir.
dataJob_expanded['minExperience_feature'] = dataJob['minExperience']
dataJob_expanded['maxExperience_feature'] = dataJob['maxExperience']

#%%

# Öncelikle, dataCv ve dataJob_expanded veri setlerindeki ilgili sütunları seçin
features_cv = dataCv[['jobseekerId', 'totalExperienceYear'] + [f'jobseekerCity_{i}' for i in range(7)] + [f'jobseekerLastPosition_{i}' for i in range(14)]]
features_job = dataJob_expanded[['jobId', 'minExperience', 'maxExperience'] + [f'jobseekerCity_{i}' for i in range(7)] + [f'jobseekerLastPosition_{i}' for i in range(14)] + ['minExperience_feature', 'maxExperience_feature']]

# Örnek olarak, dataCv'deki aday tecrübesini ve dataJob_expanded'daki tecrübe gereksinimlerini kullanarak bir uyum özelliği oluşturabiliriz
# Bu örnek, veri setlerinin birleştirilmesini gerektirmediğinden, doğrudan modelleme öncesinde kullanılabilir bilgileri gösterir

# Adayların tecrübesi (dataCv'den)
features_cv['totalExperienceYear_feature'] = features_cv['totalExperienceYear']

# İş ilanlarının tecrübe gereksinimleri (dataJob_expanded'dan)
features_job['minExperience_feature'] = features_job['minExperience']
features_job['maxExperience_feature'] = features_job['maxExperience']

# Veri setlerini doğrudan birleştiremiyoruz, ancak modelleme aşamasında bu özellikleri kullanabiliriz
# Modelinizi eğitirken, adayın tecrübesi ve işin tecrübe gereksinimlerini dikkate alarak bir uygunluk skoru hesaplayabilirsiniz

#%%

# dataAday ile dataCv'yi jobseekerId üzerinden birleştirme
merged_cv_aday = pd.merge(dataAday, dataCv, on='jobseekerId')

# Sonuç ile dataJob_expanded'ı jobId üzerinden birleştirme
final_merged_data = pd.merge(merged_cv_aday, dataJob_expanded, on='jobId')

# Örnek olarak final_merged_data'nın ilk birkaç satırını gösterelim
print(final_merged_data.head())

#%% test deneme

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Özellikleri ve hedef değişkeni belirleme
# jobId'yi hedef değişken olarak kullanıyoruz, bu örnekte basitlik adına sadece bir jobId'ye odaklanacağız.
# Gerçek uygulamada, jobId'yi one-hot encoding ile işleyebilir veya başka bir yaklaşım deneyebilirsiniz.
X = final_merged_data.drop(['jobseekerId', 'jobId', 'applicationYear', 'applicationMonth', 
                            'applicationDay', 'applicationHour', 'jobDescription_clean', 'first_word_department2'], axis=1)
y = final_merged_data['jobId'].apply(lambda x: 1 if x == 2647244 else 0)  # Örnek jobId

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli eğitme
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = rf_classifier.predict(X_test)

# Modelin performansını değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#%%

# Öncelikle, dataCv ve dataJob_expanded veri setlerini jobseekerId ve jobId üzerinden birleştirelim
# Bu örnekte, her iki DataFrame'de de jobId yok ancak, dataAday veri seti bu bağlantıyı sağlayabilir

# dataCv ve dataJob_expanded veri setlerindeki tüm jobId'leri içeren bir DataFrame oluşturalım
job_ids = dataJob_expanded[['jobId']].drop_duplicates()
jobseeker_ids = dataCv[['jobseekerId']].drop_duplicates()

# jobId ve jobseekerId için çapraz çarpım yaparak tüm mümkün kombinasyonları oluşturalım
all_combinations = jobseeker_ids.assign(key=1).merge(job_ids.assign(key=1), on='key').drop('key', axis=1)

# Oluşturulan kombinasyonları dataCv ve dataJob_expanded ile birleştirme
combined_features = all_combinations.merge(dataCv, on='jobseekerId', how='left').merge(dataJob_expanded, on='jobId', how='left')

# Şimdi, combined_features DataFrame'indeki özellikleri kullanarak model eğitimi için hazırız
print(combined_features.head())

#%%

random = final_merged_data.sample(n = 20000)

#%%

unique_jobseeker_data = final_merged_data.drop_duplicates(subset='jobseekerId', keep='first')

#%%

random = unique_jobseeker_data.sample(n = 50000)

#%%
unique_jobseeker_ids = final_merged_data['jobseekerId'].unique()

import random

n = 75000  # Oluşturmak istediğiniz örneklem büyüklüğü
sampled_jobseeker_ids = random.sample(list(unique_jobseeker_ids), n)

sampled_data = final_merged_data[final_merged_data['jobseekerId'].isin(sampled_jobseeker_ids)]


#%%
random = sampled_data.sample(n = 50000)


#%%

# jobId dışındaki tüm sütunları özellik olarak kullanın
X = random.drop(['applicationYear', 'applicationMonth', 'applicationDay','applicationHour','jobId', 'first_word_department2', 'jobDescription_clean'], axis=1)

# Hedef değişken olarak jobId kullanın
y = random['jobId']

# Hedef değişkeni kategorik bir formata dönüştürmek
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Modeli oluşturun ve eğitin
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Test seti üzerinde tahmin yapma
y_pred = rf_classifier.predict(X_test)

# Modelin performansını değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


#%%

# Modelin tahmin ettiği y_encoded değerlerini orijinal jobId'lerine dönüştürme
y_pred_jobId = label_encoder.inverse_transform(y_pred)
# Eğer jobseekerId'leri X_test setinde koruduysanız, onları kullanarak bir DataFrame oluşturun
# Örnek:
test_jobseekerIds = X_test['jobseekerId']  # jobseekerId'leri X_test'ten alın

# Tahmin edilen jobId'leri içeren bir DataFrame oluşturma
predictions_df = pd.DataFrame({
    'jobseekerId': test_jobseekerIds,
    'predictedJobId': y_pred_jobId
})

print(predictions_df)

#%% 
most_common_jobId = dataAday['jobId'].value_counts().idxmax()
count = 0
for i in dataAday['jobId']:
    if most_common_jobId == i:
        count += 1
print(count)
#%%

dataTest['jobId'] = dataTest['jobseekerId'].apply(lambda x: predictions_df[predictions_df['jobseekerId'] == x]['predictedJobId'].iloc[0] if x in predictions_df['jobseekerId'].values else most_common_jobId)
#%% 

dataTest['jobId'] = most_common_jobId

#%%
# dataTest DataFrame'ini CSV dosyası olarak kaydetme
dataTest.to_csv('dataTest_with_predictions6.csv', index=False)







