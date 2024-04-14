# JobPredictaThon

Welcome to the repository for the JobPredictaThon, a datathon competition hosted by TechCareer and Kariyer.net. This project is dedicated to processing, cleaning, and analyzing job application data and candidate details to develop a robust job recommendation system.

**Please note**: Due to competition confidentiality, we are not authorized to share the datasets used in this analysis.

## Datasets Overview

The data encompasses various facets of job applications and candidate profiles:

- **Candidate Logs**: Interaction of candidates with job listings.
- **CV Details**: Information extracted from candidate resumes.
- **Job Details**: Specifics of job listings posted by employers.
- **Test Applications**: Sample submissions by candidates.

### Detailed Data Descriptions

- `data_cv_details`:
  - `jobseekerCity`: Reflects the residing city of the candidate. "** Other" indicates candidates living abroad or those who chose not to disclose their location.
  - `totalExperienceYear`: The aggregate years of experience across all positions held, as listed in the CV.

- `data_job_details`:
  - `jobDescription`: Employer-provided descriptions requiring data cleaning for analysis.
  - `jobCity`: Lists one or more cities, indicating multi-location job vacancies.
  - `minExperience` - `maxExperience`: Range indicators for required experience levels, with specific codes for candidates with any level of experience or those without experience.

- `data_aday_log`:
  - Captures candidate applications to job postings, offering a chronological data trail for analysis.

### Scripts

- `train.py`: Initiates data loading, cleaning, and preprocessing.
- `train2.py`: Sorts candidates by application dates and cleans HTML content.
- `train3.py`: Conducts in-depth cleaning and preprocessing of text data.

## Solution Approach

The core of our project is a recommendation system tailored to match job seekers with the most suitable job postings. This system:

- Analyzes candidate preferences and historical application data.
- Employs frequency-based matching to correlate candidate attributes with job ad features.
- Prioritizes personalized job suggestions to streamline the job search process.
- Aids employers in attracting appropriate candidates efficiently.

## Getting Started

To utilize this project:

1. Clone or download this repository.
2. Install the necessary Python libraries:

```bash
pip install pandas numpy beautifulsoup4 category_encoders scikit-learn
```

Execute the provided scripts for data preprocessing and analytical processing.

## Contributing
We welcome contributions to enhance the project's effectiveness. If you're interested in contributing, please:

- Open an issue to discuss your proposed changes.
- Fork the repository and commit your contributions.
- Submit a pull request with a clear description of your improvements.
Your insights and improvements are invaluable to the evolution of this project.

---
```css
I hope this repository serves as a valuable resource in the pursuit of streamlined job matching and application processes.
```
