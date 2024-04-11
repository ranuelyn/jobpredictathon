#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

#%% rank column

# applicationDate'e göre sıralama yapılır ve her aday için rank hesaplanır
dataAday['applicationDate'] = pd.to_datetime(dataAday['applicationDate'])
dataAday['rank'] = dataAday.groupby('jobseekerId')['applicationDate'].rank(method='first', ascending=True)

# Kontrol için ilk 5 satırı göster
print(dataAday.head())

#%% html parse

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# jobDescription sütununu temizle
dataJob['jobDescription'] = dataJob['jobDescription'].apply(clean_html)

# Gereksiz karakterleri kaldır
dataJob['jobDescription'] = dataJob['jobDescription'].apply(lambda x: re.sub(r'\n', ' ', x))

# Kontrol için ilk 5 satırı göster
print(dataJob['jobDescription'].head())

#%% Drop "Unnamed" Column

dataAday.drop('Unnamed: 0', axis=1, inplace=True)
dataCv.drop('Unnamed: 0', axis=1, inplace=True)
dataJob.drop('Unnamed: 0', axis=1, inplace=True)

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

#%%

#%% add the columns

# Orijinal jobseekerCity sütununu kaldırma
dataCv.drop('jobseekerCity', axis=1, inplace=True)

# İkili kodlanmış sütunları dataCv veri setine ekleyin
dataCv = pd.concat([dataCv, dataCv_encoded], axis=1)

# Orijinal jobCity sütununu kaldırma
dataJob_expanded.drop('jobseekerCity', axis=1, inplace=True)

# İkili kodlanmış sütunları dataJob veri setine ekleyin
dataJob_expanded = pd.concat([dataJob_expanded, dataJob_encoded], axis=1)

#%%
# Her aday için en son 2 başvurunun test, geri kalanların train seti olarak ayrılması
dataAday['set'] = 'train'
dataAday.loc[dataAday.groupby('jobseekerId')['rank'].nlargest(2).index.get_level_values(1), 'set'] = 'test'

#%%

# Örnek birleştirme işlemi
# Öncelikle, data_aday_log ve dataCv_encoded verilerini jobseekerId üzerinden birleştir
merged_df = pd.merge(dataAday, dataCv, on='jobseekerId', how='left')

# Daha sonra, merged_df ve dataJob_encoded verilerini jobId üzerinden birleştir
merged_df = pd.merge(merged_df, dataJob_expanded, on='jobId', how='left')

# Kontrol için ilk 5 satırı göster
print(merged_df.head())

#%%

# Özelliklerin veri tiplerini optimize etme
for col in merged_df.select_dtypes(include=['float64', 'int64']).columns:
    col_min = merged_df[col].min()
    col_max = merged_df[col].max()
    if pd.api.types.is_float_dtype(merged_df[col]):
        merged_df[col] = pd.to_numeric(merged_df[col], downcast='float')
    else:
        if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
            merged_df[col] = pd.to_numeric(merged_df[col], downcast='integer')
        elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
            merged_df[col] = pd.to_numeric(merged_df[col], downcast='integer')

#%%
# Sayısal sütunlar için veri tipi optimizasyonu
for col in ['jobseekerId', 'jobId']:
    merged_df[col] = pd.to_numeric(merged_df[col], downcast='integer')

for col in merged_df.columns:
    if 'jobseekerCity' in col:
        merged_df[col] = merged_df[col].astype('int8')


#%% test
# Öncelikle, 'set' sütununa dayanarak mevcut test ve train ayrımını kaldırıyoruz
merged_df.drop('set', axis=1, inplace=True)

# Sonra, her kullanıcı için son 1 başvuruyu rastgele test olarak seçelim
merged_df['set'] = 'train'  # Varsayılan olarak her şeyi train olarak ayarla
merged_df.loc[merged_df.groupby('jobseekerId').sample(n=1, random_state=42).index, 'set'] = 'test'

# Test seti
test_samples = merged_df[merged_df['set'] == 'test']

# Train setinden rastgele örnekler seçme
# Bu sefer, her kullanıcı için belirli bir sayıda (örneğin, N=5) train verisi seçilecek
N = 1  # Her kullanıcı için seçilecek train verisi sayısı
train_samples = merged_df[merged_df['set'] == 'train'].groupby('jobseekerId').sample(n=N, replace=True, random_state=42)

# Test ve train örneklerinin birleştirilmesi
sampled_df = pd.concat([test_samples, train_samples]).reset_index(drop=True)

# Birleştirilmiş veri setinin boyutunu kontrol etme
print(f"Sampled DataFrame shape: {sampled_df.shape}")


#%%

# Hedef değişken ve özelliklerin ayrılması
X = sampled_df.drop(['jobId', 'set', 'jobseekerLastPosition', 'departmentName'], axis=1)  # 'set' ve 'jobId' sütunları hariç tutulur
y = sampled_df['jobId']  # Hedef değişken

# Model için gereksiz olan veya metin işleme gerektiren sütunları çıkarabilirsiniz
# Örneğin, jobDescription metin sütunu burada çıkarılmıştır
X = X.drop(['jobDescription', 'applicationDate', 'jobPosition'], axis=1)  # Metin sütunu ve tarih sütunu çıkarılır

#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri setini eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Logistic Regression modelinin oluşturulması ve eğitimi
log_reg_model = LogisticRegression(max_iter=100, solver='liblinear', random_state=42)
log_reg_model.fit(X_train, y_train)

# Modelin test seti üzerindeki performansının değerlendirilmesi
y_pred_log_reg = log_reg_model.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Model Doğruluğu: {accuracy_log_reg}")



























