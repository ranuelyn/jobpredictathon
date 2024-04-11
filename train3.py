#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 02:01:01 2024

@author: ranuelyn
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 01:34:49 2024

@author: yusuf
"""

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

#%%
#%% html parse

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# jobDescription sütununu temizle
dataJob['jobDescription'] = dataJob['jobDescription'].apply(clean_html)

# Gereksiz karakterleri kaldır
dataJob['jobDescription'] = dataJob['jobDescription'].apply(lambda x: re.sub(r'\n', ' ', x))
dataJob['jobDescription'] = dataJob['jobDescription'].apply(lambda x: re.sub(r'\s+', ' ', x).strip()) # Birden fazla boşlukları temizle

# Küçük harfe dönüştür
dataJob['jobDescription'] = dataJob['jobDescription'].apply(lambda x: x.lower())

# Kontrol için ilk 5 satırı göster
print(dataJob['jobDescription'].head())
#%%
dataCv['departmentName'] = dataCv['departmentName'].str.split(r'\s|/').str[0]
#%%
dataCv['departmentName'] = dataCv['departmentName'].apply(lambda x: x.lower())
#%%
# İlk olarak, her şehri ayrı bir liste elemanı olarak ayırma
expanded_jobCity = dataJob['jobCity'].str.split(',', expand=True)

# Şimdi, bu genişletilmiş DataFrame'i orijinal DataFrame'e satır olarak ekleyin
# Bu işlem, her bir şehri ayrı bir satır olarak ele alacak
expanded_jobCity = expanded_jobCity.stack().reset_index(level=1, drop=True).rename('jobCity')

# Orijinal DataFrame'e bu yeni şehir sütunu ekleyin
dataJob_expanded = dataJob.drop('jobCity', axis=1).join(expanded_jobCity)
dataJob = dataJob_expanded
# Sonuçları kontrol edin
print(dataJob_expanded.head())
#%% rank column

# applicationDate'e göre sıralama yapılır ve her aday için rank hesaplanır
dataAday['applicationDate'] = pd.to_datetime(dataAday['applicationDate'])
dataAday['rank'] = dataAday.groupby('jobseekerId')['applicationDate'].rank(method='first', ascending=True)

# Kontrol için ilk 5 satırı göster
print(dataAday.head())

#%%
# dataAday ve dataCv veri çerçevelerini jobseekerId üzerinden birleştir
merged_data = pd.merge(dataAday, dataCv, on='jobseekerId')

# İlanlara başvuru sayısını hesaplamak için jobId ve jobseekerCity'e göre gruplama yapılır ve sayımlar hesaplanır
city_job_applications = merged_data.groupby(['jobseekerCity', 'jobId']).size().reset_index(name='applicationCount')

# Her şehir için başvuru sayısına göre azalan sırada sıralama yapılır
city_job_applications_sorted = city_job_applications.sort_values(['jobseekerCity', 'applicationCount'], ascending=[True, False])

# Her şehirdeki ilanlara göre rank hesaplanır
city_job_applications_sorted['rank'] = city_job_applications_sorted.groupby('jobseekerCity')['applicationCount'].rank(method='first', ascending=False)

# Kontrol için ilk 5 satırı göster
print(city_job_applications_sorted.head())

#%%

# dataAday ve dataCv veri çerçevelerini jobseekerId üzerinden birleştir
merged_data = pd.merge(dataAday, dataCv[['jobseekerId', 'jobseekerCity']], on='jobseekerId', how='left')

# Kontrol için ilk 5 satırı göster
print(merged_data.head())

#%%

# Her bir şehir için en çok başvuru yapılan ilanları bul
city_popular_jobs = merged_data.groupby(['jobseekerCity', 'jobId']).size().reset_index(name='applicationCount')
city_popular_jobs = city_popular_jobs.sort_values(['jobseekerCity', 'applicationCount'], ascending=[True, False])

#%%

# dataTest veri setini güncelleme
for index, row in dataTest.iterrows():
    jobseeker_id = row['jobseekerId']
    
    # Bu jobseekerId için merged_data'dan adayın şehrini ve başvurduğu iş ilanlarını bulma
    applicant_info = merged_data[merged_data['jobseekerId'] == jobseeker_id]
    if not applicant_info.empty:
        applicant_city = applicant_info['jobseekerCity'].iloc[0]
        applied_jobs = applicant_info['jobId'].unique()
        
        # Bu şehir için city_job_applications_sorted içerisinde adayın daha önce başvurmadığı en popüler ilanları bulma
        city_jobs = city_job_applications_sorted[(city_job_applications_sorted['jobseekerCity'] == applicant_city) & (~city_job_applications_sorted['jobId'].isin(applied_jobs))]
        
        if not city_jobs.empty:
            # Adayın daha önce başvurmadığı en popüler ilanı seçme
            # En popüler ilan, applicationCount'a göre sıralandığında en üstte olan ilk ilandır
            top_job_for_city = city_jobs.sort_values(by='applicationCount', ascending=False).iloc[0]['jobId']
        else:
            # Eğer adayın başvurmadığı uygun bir ilan kalmadıysa, placeholder kullan
            top_job_for_city = 'No new suitable job found'
    else:
        top_job_for_city = 'Applicant city not found'
    
    # Bulunan jobId'yi dataTest veri setindeki ilgili satıra yazma
    dataTest.at[index, 'jobId'] = top_job_for_city

# Güncellenmiş dataTest veri setini göster
print(dataTest.head())
#%%
# merged_data veri çerçevesini oluştururken adayların tecrübe bilgilerini de ekleyelim
# Bu örnekte, dataCv'de adayların tecrübe bilgilerinin olduğunu varsayıyoruz
merged_data = pd.merge(merged_data, dataCv[['jobseekerId', 'totalExperienceYear']], on='jobseekerId', how='left')

#%%
def is_experience_matching(applicant_experience, min_exp, max_exp):
    # Tecrübe gereksinimi 99-99 ise her türlü tecrübe kabul edilir
    if min_exp == 99 and max_exp == 99:
        return True
    # Tecrübe gereksinimi 98-98 ise sadece tecrübesiz adaylar kabul edilir
    elif min_exp == 98 and max_exp == 98:
        return applicant_experience == 0
    # Min tecrübe gereksinimi 0 ve max bir değer ise, bu max değere kadar tecrübeli adaylar kabul edilir
    elif min_exp == 0 and max_exp > 0:
        return applicant_experience <= max_exp
    # Min tecrübe gereksinimi bir değer ve max 0 ise, bu min değerden fazla tecrübeli adaylar kabul edilir
    elif min_exp > 0 and max_exp == 0:
        return applicant_experience >= min_exp
    # Diğer durumlarda, adayın tecrübesi belirtilen aralıkta olmalıdır
    else:
        return min_exp <= applicant_experience <= max_exp


#%%
for index, row in dataTest.iterrows():
    jobseeker_id = row['jobseekerId']
    
    # Adayın şehri, tecrübesi, son pozisyonu ve başvurduğu işler
    if jobseeker_id in merged_data['jobseekerId'].values:
        applicant_info = dataCv[dataCv['jobseekerId'] == jobseeker_id].iloc[0]
        applicant_city = applicant_info['jobseekerCity']
        applicant_experience = applicant_info['totalExperienceYear']
        applicant_last_position = applicant_info['jobseekerLastPosition']
        applied_jobs = merged_data[merged_data['jobseekerId'] == jobseeker_id]['jobId'].unique()
        
        # Adayın şehri ve pozisyonuyla eşleşen işler
        matching_jobs = dataJob[(dataJob['jobCity'] == applicant_city) & (dataJob['jobPosition'] == applicant_last_position) & (~dataJob['jobId'].isin(applied_jobs))]
        
        # Eşleşen işlerden tecrübeye uyan en popüler işi bulma
        found_job = False
        for _, job_row in matching_jobs.iterrows():
            job_id = job_row['jobId']
            job_min_exp, job_max_exp = job_row[['minExperience', 'maxExperience']]
            
            if is_experience_matching(applicant_experience, job_min_exp, job_max_exp):
                dataTest.at[index, 'jobId'] = job_id
                found_job = True
                break
        
        # Eğer adayın pozisyonu ve şehriyle eşleşen uygun iş bulunamazsa, daha genel kriterleri kullan
        if not found_job:
            # Bu şehir için en popüler işler ve adayın başvurmadığı işler
            city_jobs = city_job_applications_sorted[(city_job_applications_sorted['jobseekerCity'] == applicant_city) & (~city_job_applications_sorted['jobId'].isin(applied_jobs))]

            for _, job_row in city_jobs.iterrows():
                job_id = job_row['jobId']
                job_min_exp, job_max_exp = dataJob.loc[dataJob['jobId'] == job_id, ['minExperience', 'maxExperience']].values[0]
                
                if is_experience_matching(applicant_experience, job_min_exp, job_max_exp):
                    dataTest.at[index, 'jobId'] = job_id
                    found_job = True
                    break
            
            # Eğer şehirde adayın tecrübesine uygun ve daha önce başvurmadığı bir iş bulunamazsa, en popüler işi atama
            if not found_job and not city_jobs.empty:
                dataTest.at[index, 'jobId'] = city_jobs.iloc[0]['jobId']

    else:
        dataTest.at[index, 'jobId'] = 'Applicant info not found'

# Güncellenmiş dataTest veri setini göster
print(dataTest)
#%%
def check_department_in_description(department, description):
    # departmentName içindeki kelimelerin jobDescription içinde geçip geçmediğini kontrol et
    return any(word.lower() in description.lower() for word in department.split())


#%% test3
for index, row in dataTest.iterrows():
    jobseeker_id = row['jobseekerId']
    
    # Aday bilgilerini çek
    if jobseeker_id in merged_data['jobseekerId'].values:
        applicant_info = dataCv[dataCv['jobseekerId'] == jobseeker_id].iloc[0]
        applicant_city = applicant_info['jobseekerCity']
        applicant_experience = applicant_info['totalExperienceYear']
        applicant_last_position = applicant_info['jobseekerLastPosition']
        applicant_department = applicant_info['departmentName']
        applied_jobs = merged_data[merged_data['jobseekerId'] == jobseeker_id]['jobId'].unique()

        # Adayın şehri, departmanı ve pozisyonuyla eşleşen iş ilanlarını filtrele
        matching_jobs = dataJob[
            (dataJob['jobCity'] == applicant_city) & 
            (~dataJob['jobId'].isin(applied_jobs))
        ]
        
        # Departman, pozisyon ve tecrübe uygunluğuna göre iş ilanlarını filtrele
        department_position_exp_match = matching_jobs[
            matching_jobs['jobDescription'].apply(lambda desc: check_department_in_description(applicant_department, desc)) &
            (matching_jobs['jobPosition'] == applicant_last_position)
        ]

        # Uygun iş ilanlarını tecrübe ve popülerliğe göre sırala
        if not department_position_exp_match.empty:
            selected_job = department_position_exp_match.merge(city_job_applications_sorted, on="jobId").sort_values(by="applicationCount", ascending=False)
            selected_job = selected_job[selected_job.apply(lambda x: is_experience_matching(applicant_experience, x['minExperience'], x['maxExperience']), axis=1)]
            if not selected_job.empty:
                dataTest.at[index, 'jobId'] = selected_job.iloc[0]['jobId']
                print(f"Aday {index + 1} / jobseekerId: {jobseeker_id}, seçilen jobId: {selected_job.iloc[0]['jobId']}")
                continue

        # Eğer departman ve pozisyon uygun iş bulunamazsa, genel kriterlere göre iş ara
        popular_jobs_in_city = city_job_applications_sorted[(city_job_applications_sorted['jobseekerCity'] == applicant_city) & (~city_job_applications_sorted['jobId'].isin(applied_jobs))]
        if not popular_jobs_in_city.empty:
            dataTest.at[index, 'jobId'] = popular_jobs_in_city.iloc[0]['jobId']
    else:
        dataTest.at[index, 'jobId'] = 'Applicant info not found'
        print(f"Aday {index + 1} / jobseekerId: {jobseeker_id}, bilgi bulunamadı")

# jobId sütununu int'e çevir
dataTest['jobId'] = dataTest['jobId'].astype(float).astype('Int64')

# Güncellenmiş dataTest veri setini göster
print(dataTest.head())
#%% test4

for index, row in dataTest.iterrows():
    jobseeker_id = row['jobseekerId']
    
    # Adayın bilgilerini çek
    if jobseeker_id in merged_data['jobseekerId'].values:
        applicant_info = dataCv[dataCv['jobseekerId'] == jobseeker_id].iloc[0]
        applicant_city = applicant_info['jobseekerCity']
        applicant_experience = applicant_info['totalExperienceYear']
        applicant_last_position = applicant_info['jobseekerLastPosition']
        applicant_department = applicant_info['departmentName'].lower()
        applied_jobs = merged_data[merged_data['jobseekerId'] == jobseeker_id]['jobId'].unique()

        # En popüler işleri adayın şehrine göre filtrele
        popular_jobs_in_city = city_job_applications_sorted[(city_job_applications_sorted['jobseekerCity'] == applicant_city) & (~city_job_applications_sorted['jobId'].isin(applied_jobs))]
        
        # Filtrelenmiş işler içinde, departman ve pozisyona uyanları bul
        found_job = False
        for _, job_row in popular_jobs_in_city.iterrows():
            job_id = job_row['jobId']
            if job_id in applied_jobs:
                continue
                
            job_details = dataJob.loc[dataJob['jobId'] == job_id].iloc[0]
            job_description = job_details['jobDescription'].lower()
            job_position = job_details['jobPosition']
            job_min_exp, job_max_exp = job_details['minExperience'], job_details['maxExperience']

            # Departman uyumluluğu ve pozisyon eşleşmesi kontrolü
            if (applicant_department in job_description or job_position == applicant_last_position) and is_experience_matching(applicant_experience, job_min_exp, job_max_exp):
                dataTest.at[index, 'jobId'] = job_id
                found_job = True
                print(f"İşlem No: {index + 1}, jobseekerId: {jobseeker_id}, Seçilen jobId: {job_id}")
                break
        
        # Eğer departman ve pozisyona uygun iş bulunamazsa, şehirdeki en popüler işe bak
        if not found_job:
            # Adayın daha önce başvurmadığı ve şehrindeki en popüler iş ilanlarını filtrele
            popular_jobs_in_city_unapplied = popular_jobs_in_city[~popular_jobs_in_city['jobId'].isin(applied_jobs)]
    
            if not popular_jobs_in_city_unapplied.empty:
                # Adayın daha önce başvurmadığı en popüler işi seç
                top_job_for_city_unapplied = popular_jobs_in_city_unapplied.iloc[0]['jobId']
                dataTest.at[index, 'jobId'] = top_job_for_city_unapplied
                print(f"İşlem No: {index + 1}, jobseekerId: {jobseeker_id}, Genel En Popüler jobId (Daha Önce Başvurulmamış): {top_job_for_city_unapplied}")
            else:
                # Eğer adayın daha önce başvurmadığı ve şehrinde uygun bir iş bulunamazsa
                dataTest.at[index, 'jobId'] = 'No new suitable job found'
                print(f"İşlem No: {index + 1}, jobseekerId: {jobseeker_id}, Uygun Yeni İş Bulunamadı")

    else:
        dataTest.at[index, 'jobId'] = 'Applicant info not found'
        print(f"İşlem No: {index + 1}, jobseekerId: {jobseeker_id}, Bilgi Bulunamadı")

# jobId sütununu float'tan int'e çevirme
dataTest['jobId'] = dataTest['jobId'].astype(float).astype('Int64')

# Güncellenmiş dataTest veri setini göster
print(dataTest.head())



#%%
dataTest.to_csv('dataTest_4.csv', index=False)
