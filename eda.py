import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import cohen_kappa_score

# Load the entire dataset from a JSON file
with open("/Users/jamesgoldbach/CS383/GroupProject/medical_rag_dataset_v2.json", "r") as file:
    data = json.load(file)

# Convert case data to DataFrame
cases = pd.json_normalize(data['cases'])
print(f"Dataset shape: {cases.shape}")  # Display the shape of the dataset, should be (1000, columns)

# Step 3: Statistics about dataset attributes
stats = cases.describe(include='all')
print(stats)

# Visualization of Age distribution
cases['patient_age'] = pd.to_numeric(cases['patient_profile.age'], errors='coerce')

plt.figure(figsize=(10, 6))
sns.histplot(cases['patient_age'].dropna(), bins=20, kde=True, color='blue')
plt.title('Distribution of Patient Age')
plt.xlabel('Age')  # Correct usage for xlabel
plt.ylabel('Frequency')  # Correct usage for ylabel
plt.show()

# Visualization of Gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='patient_profile.gender', data=cases)
plt.title('Gender Distribution')
plt.xlabel('Gender')  # Correct usage for xlabel
plt.ylabel('Count')  # Correct usage for ylabel
plt.show()

# Visualization of Chief Complaints
chief_complaints = cases['presentation.chief_complaint'].explode()
plt.figure(figsize=(12, 6))
sns.countplot(y=chief_complaints)
plt.title('Chief Complaints Frequency')
plt.ylabel('Complaints')  # Correct usage for ylabel
plt.xlabel('Count')  # Correct usage for xlabel
plt.show()

# WordCloud for Medical History
medical_history = ' '.join(cases['patient_profile.medical_history'].explode().dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(medical_history)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis
plt.title('Word Cloud of Medical History')
plt.show()

# Step 4: Comment on Findings
# Write your observations below.
'''
- The age distribution shows that there is a predominance of elderly patients, with patients predominantly above 65 years.
- The gender distribution indicates that there is a balance between male and female patients, with some cases where gender is marked as non-binary (NB).
- Chief complaints primarily involve polyuria and diarrhea, which is essential for understanding the high prevalence of gastrointestinal conditions in these patients.
- The word cloud reveals that conditions like Celiac Disease and Crohn's Disease frequently appear in medical histories, which suggests a trend or bias towards gastrointestinal disorders among the sampled patients.
'''

# Step 5: Inter-annotator metrics for labeling (if applicable)
# Uncomment and modify the following lines to include actual annotations if available.
'''
annotations = pd.DataFrame({
    'annotator1': ['Condition A', 'Condition B', 'Condition A'],  # Example data
    'annotator2': ['Condition A', 'Condition C', 'Condition A']   # Example data
})
kappa = cohen_kappa_score(annotations['annotator1'], annotations['annotator2'])
print('Cohen\'s Kappa:', kappa)
'''

