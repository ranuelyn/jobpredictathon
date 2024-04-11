# JobPredictaThon
TechCareer &amp; Kariyet.net Datathon: JobPredictaThon

Bu repo, TechCareer ve Kariyer.net tarafından düzenlenen bir datathon yarışmasında kullanılan kodları açıklar. Projede, iş başvuruları ve aday detayları üzerinde ön işleme, temizleme ve analiz işlemleri gerçekleştirilmiştir.

# Veri Setleri

Bu projede kullanılan veri setleri aşağıdaki bilgileri içerir:

Aday logları
CV detayları
İş detayları
Test başvuruları
Verilen veriler hakkında açıklamalar:
data_cv_details icin:
jobseekerCity:

Adayın yaşadığı şehri göstermekte olup, “** Diğer” seçmiş olanlar, yurt dışında ikamet eden (Kıbrıs hariç. Kıbrıs için Girne, Lefkoşa vb. il bazında kayıtlar görebilirsiniz) veya bu bilgiyi paylaşmak istememiş olan adaylardır.

totalExperienceYear:

Adayın sadece o pozisyonda değil, bu cv’si özelinde tüm iş deneyimlerinin toplam yılını temsil etmektedir. totalExperienceYear’ ı 0 olan adayların toplam tecrübesi 12 aydan az demektir.

data_job_details için:
jobDescription:

İşverenlerin girmiş oldukları free text alandır. Bu bakımdan, kullanım öncesi temizleme işlemleri gerekmektedir.

jobCity:

Birden fazla il varsa, o ilan için birden fazla ilde çalışacak adaylar arıyorlar demektir.

minExperience - maxExperience için örnekler üzerinden açıklamalar

5 - 0 : en az 5 yıl tecrübesi olan adaylar.
0 - 5 : en çok 5 yıl tecrübeli adaylar.
99 - 99 : tecrübeli ya da tecrübesiz adaylar.
98 - 98 : tecrübesiz adaylar.
Bunlar dışındakilerin hepsi, o aralıktaki tecrübeyi göstermektedir.
5 - 9 : en az 5 yıl, en çok 9 yıl tecrübeli adaylar.
şeklinde.

data_aday_log için:
Adaylar ve o adayların hangi tarihlerde hangi ilanlara başvuru yaptıkları bilgileri paylaşılmıştır. Train setinizi isterseniz tamamından, isterseniz dilediğiniz mantığa göre oluşturduğunuz bir aralıktan alabilirsiniz.

train.py: Veri yükleme, temizleme ve ön işleme adımlarını içeren Python scripti.
train2.py: Adayların başvuru tarihlerine göre sıralanması ve HTML içeriğinin temizlenmesi gibi işlemleri içeren Python scripti.
train3.py: Metin verisi üzerinde daha detaylı temizleme ve ön işleme işlemlerini gerçekleştiren Python scripti.

Katkıda Bulunma

Projeye katkıda bulunmak isterseniz, lütfen öncelikle projenin nasıl geliştirilebileceğine dair fikirlerinizi ya da planladığınız değişiklikleri tartışmak üzere bir issue açınız.

