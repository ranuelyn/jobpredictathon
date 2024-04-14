## JobPredictaThon
TechCareer &amp; Kariyet.net Datathon: JobPredictaThon
(I am not authorized to share data due to competition confidentiality.)

This repository explains the codes used in a datathon competition organized by TechCareer and Kariyer.net. The project involves preprocessing, cleaning, and analyzing data on job applications and candidate details.

## Datasets

The datasets used in this project include information on:

Candidate logs
CV details
Job details
Test applications
Descriptions for the provided data:
For data_cv_details:
jobseekerCity:

Indicates the city the candidate lives in. Candidates who have chosen "** Other" reside outside the country (excluding Cyprus. You may see records based on the city such as Girne, Lefkoşa for Cyprus) or do not wish to share this information.

totalExperienceYear:

Represents the total years of experience of the candidate not just in that position, but across all job experiences in that CV. Candidates with a totalExperienceYear of 0 have less than 12 months of total experience.

For data_job_details:
jobDescription:

A free text field entered by employers. Therefore, it requires cleaning before use.

jobCity:

If there is more than one city listed, it means they are looking for candidates to work in multiple cities for that advertisement.

Explanations for minExperience - maxExperience through examples:

- 5 - 0: Candidates with at least 5 years of experience.
- 0 - 5: Candidates with up to 5 years of experience.
- 99 - 99: Candidates with or without experience.
- 98 - 98: Candidates without experience.
- All others show the range of experience required.
For example, 5 - 9: Candidates with at least 5 years, up to 9 years of experience.
For data_aday_log:
Information on the candidates and which advertisements they have applied to on specific dates is shared. You can create your training set from the entirety or any subset based on the logic you prefer.

train.py: Python script that includes data loading, cleaning, and preprocessing steps.
train2.py: Python script that includes sorting candidates by application dates and cleaning HTML content.
train3.py: Python script that performs more detailed cleaning and preprocessing on text data.

## Solution Approach

The solution method adopted in the project is to develop a recommendation system that suggests the most suitable job advertisements based on candidates' preferences and past applications. This recommendation system analyzes the characteristics of candidates and job advertisements, performing a frequency-based comparison. The system calculates the frequency of characteristics in advertisements that candidates have applied to or shown interest in the past and compares these details with new and existing job advertisements to make suggestions. This method facilitates the job-finding process for candidates by providing personalized job recommendations and helps employers reach the right candidates.

## Setup

Steps to use the project:

- Clone or download the repo.
- Install the required Python libraries:
- pip install pandas numpy beautifulsoup4 category_encoders scikit-learn
- Run the scripts to perform data preprocessing and analysis.

## Contributing
If you would like to contribute to the project, please first discuss your ideas or the changes you plan to make by opening an issue.
