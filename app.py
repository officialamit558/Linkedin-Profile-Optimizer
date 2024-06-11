import streamlit as st
import http.client
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.fillna('', inplace=True)  # Fill missing values
    return df

# Concatenate relevant fields based on job role
def concatenate_fields(df, job_role=None):
    if job_role:
        relevant_df = df[df['Job Role'] == job_role]
    else:
        relevant_df = df
    return relevant_df['Projects'] + " " + relevant_df['Experience'] + " " + relevant_df['Bio'] + " " + relevant_df['About'] + " " + relevant_df['Skills']

# Feature extraction
def extract_features(df, vectorizer=None, job_role=None):
    combined_text = concatenate_fields(df, job_role)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=100)
        features = vectorizer.fit_transform(combined_text).toarray()
    else:
        features = vectorizer.transform(combined_text).toarray()

    return features, vectorizer

# Calculate cosine similarity
def calculate_similarity(features, profile_features):
    similarities = cosine_similarity(features, profile_features)
    return similarities

# Define industry standard skills for each role
industry_standard_skills = {
    'AI Engineer': {'python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision'},
    'Frontend Developer': {'javascript', 'react', 'html', 'css', 'typescript', 'redux', 'angular'},
    'Software Engineer': {'java', 'python', 'c++', 'data structures', 'algorithms', 'software development', 'oop'},
    'SDE': {'java', 'c++', 'python', 'data structures', 'algorithms', 'system design', 'oop'},
    'Backend Developer': {'java', 'python', 'node.js', 'databases', 'sql', 'api', 'microservices'},
    'DevOps Engineer': {'docker', 'kubernetes', 'aws', 'azure', 'ci/cd', 'terraform', 'linux'},
    'Data Scientist': {'python', 'r', 'statistics', 'machine learning', 'data analysis', 'pandas', 'numpy'},
    'ML Engineer': {'python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data engineering'}
}

# Recommendation system
def generate_recommendations(profile, ideal_profiles, features, vectorizer, job_role=None):
    profile_df = pd.DataFrame([profile])
    profile_features, _ = extract_features(profile_df, vectorizer, job_role)
    similarities = calculate_similarity(features, profile_features)

    score = np.mean(similarities) * 10  # Scale the similarity score to a 0-10 range
    recommendations = []

    # Detailed criteria evaluation
    if score < 7:
        if not profile['Bio']:
            recommendations.append('Add a detailed bio to highlight your professional summary.')
        if len(profile['Skills'].split(',')) < 5:
            recommendations.append('Add more relevant skills to showcase your expertise.')
        if len(profile['Experience'].split(',')) < 3:
            recommendations.append('Add more job experiences to demonstrate your career history.')
        if not profile['Projects']:
            recommendations.append('Include significant projects to highlight your practical experience.')

        # Specific recommendations based on missing elements
        if not profile['About']:
            recommendations.append('Include an "About" section to provide a detailed personal narrative.')
        if 'leadership' not in profile['Skills'].lower():
            recommendations.append('Consider adding leadership skills if applicable.')
        if 'teamwork' not in profile['Skills'].lower():
            recommendations.append('Highlight teamwork skills if relevant to your job role.')
        if 'communication' not in profile['Skills'].lower():
            recommendations.append('Emphasize communication skills, which are essential for most roles.')

        
        # Industry standards comparison
        standard_skills = industry_standard_skills.get(job_role, set())
        profile_skills = set(skill.strip().lower() for skill in profile['Skills'].split(','))
        missing_skills = standard_skills - profile_skills
        if missing_skills:
            recommendations.append(f'Consider adding these skills: {", ".join(missing_skills)}')


    return score, recommendations

# Check missing skills
def get_missing_skills(profile_skills, job_role_skills):
    profile_skills_set = set(skill.strip().lower() for skill in profile_skills.split(','))
    job_role_skills_set = set(skill.strip().lower() for skill in job_role_skills)
    missing_skills = job_role_skills_set - profile_skills_set
    return missing_skills

# Fetch profile data from LinkedIn
# Fetch profile data from LinkedIn
def fetch_linkedin_profile(linkedin_url, api_key):
    conn = http.client.HTTPSConnection("linkedin-data-api.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "linkedin-data-api.p.rapidapi.com"
    }
    conn.request("GET", f"/get-profile-data-by-url?url={linkedin_url}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))


# Main Streamlit app
def main():
    # Customize the appearance
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f3f2ef;
            color: #004182;
        }
        .stButton>button {
            background-color: #0073b1;
            color: #ffffff;
        }
        .stSelectbox, .stTextInput {
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("LinkedIn Profile Optimizer")
    
    filepath = 'linkedin_optimisation.csv'  # Path to your dataset
    df = load_data(filepath)
    
    job_roles = [
        "AI Engineer", "Frontend Developer", "Software Engineer", "SDE",
        "Backend Developer", "DevOps Engineer", "Data Scientist", "ML Engineer"
    ]
    
    job_role = st.selectbox("Select your job role", job_roles)
    linkedin_url = st.text_input("Enter your LinkedIn profile URL")
    api_key = "2f2963ea7bmshc94310edd911140p131777jsnc87883611adf"

    if st.button("Optimize Profile"):
        profile_data = fetch_linkedin_profile(linkedin_url , api_key)
        
        new_profile = {
            'Job Role': job_role,
            'Projects': '',  # Projects data is not available in the provided API response
            'Experience': '',  # Experience data is not explicitly available in the provided API response
            'Bio': profile_data.get('headline', ''),
            'About': profile_data.get('summary', ''),
            'Skills': ', '.join([skill['name'] for skill in profile_data.get('skills', [])])
        }

        features, vectorizer = extract_features(df, job_role=job_role)
        score, recommendations = generate_recommendations(new_profile, df, features, vectorizer, job_role)

        job_role_skills = df[df['Job Role'] == job_role]['Skills']
        job_role_skills_list = ', '.join(job_role_skills).split(', ')

        missing_skills = get_missing_skills(new_profile['Skills'], job_role_skills_list)
        if missing_skills:
            recommendations.append(f'Consider adding these skills: {", ".join(missing_skills)}')

        st.write(f'Score: {score}/10')
        st.write('Recommendations:')
        for recommendation in recommendations:
            st.write(f"- {recommendation}")

if __name__ == "__main__":
    main()
