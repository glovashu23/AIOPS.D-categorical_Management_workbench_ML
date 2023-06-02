import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)


# Read benchmark data from CSV file
benchmark_data = pd.read_csv('/Users/agurmu/Desktop/AIOBS/cat_management/benchmark.csv')

# Specify query information
query_department = 'Account Management'
query_role_title = 'Associate Director, Strategy'
query_role_description = 'The Associate Director, Strategy supports the strategy team in developing and executing strategic initiatives. They conduct research, analyze data, provide insights, assist in strategic planning, collaborate with internal stakeholders, and contribute to business growth and success.'

# Set top-k and model name for Sentence Transformer
top_k = 10
model_name = 'all-MiniLM-L6-v2'

# Data preprocessing for benchmark data
benchmark_data['Department'] = benchmark_data['Department'].apply(preprocess_text)
benchmark_data['Role Title'] = benchmark_data['Role Title'].apply(preprocess_text)
benchmark_data['Role Description'] = benchmark_data['Role Description'].apply(preprocess_text)

# Load pre-trained Sentence Transformer model
model = SentenceTransformer(model_name)

# Create a DataFrame to store the results for all suppliers
all_supplier_results = pd.DataFrame(columns=[
    'Supplier Type', 'Supplier Department', 'Supplier Role Title', 'Supplier Role Description',
    'Benchmark Department', 'Benchmark Role Title', 'Benchmark Role Description', 'Benchmark Avg Rate',
    'Confidence Score', 'Fully Loaded Rate', 'Rate Difference', 'Years of Experience', 'Estimated Hours',
    'Currency', 'Total Rate'
])

# Iterate over each supplier
for supplier_id in range(1, 4):  # Modify the range as per the number of suppliers
    # Read supplier data from CSV file
    supplier_data = pd.read_csv(f'/Users/agurmu/Desktop/AIOBS/cat_management/supplier{supplier_id}.csv')

    # Data preprocessing for supplier data
    supplier_data['Department'] = supplier_data['Department'].apply(preprocess_text)
    supplier_data['Role Title'] = supplier_data['Role Title'].apply(preprocess_text)
    supplier_data['Role Description'] = supplier_data['Role Description'].apply(preprocess_text)

    # Combine text data for embedding
    benchmark_data['Text'] = benchmark_data['Department'] + ' ' + benchmark_data['Role Title'] + ' ' + benchmark_data[
        'Role Description']
    supplier_data['Text'] = supplier_data['Department'] + ' ' + supplier_data['Role Title'] + ' ' + supplier_data[
        'Role Description']

    # Convert text data to embeddings
    benchmark_embeddings = model.encode(benchmark_data['Text'].tolist())
    supplier_embeddings = model.encode(supplier_data['Text'].tolist())

   
    # Calculate similarity scores using cosine similarity
    similarity_scores = cosine_similarity(supplier_embeddings, benchmark_embeddings)

    # Find the top-k benchmark matches for each supplier role
for i in range(similarity_scores.shape[0]):
    supplier_row = supplier_data.iloc[i]
    top_k_indices = similarity_scores[i].argsort()[-top_k:][::-1]  # Get indices of top-k similarity scores

    for idx in top_k_indices:
        benchmark_row = benchmark_data.iloc[idx]

        # Calculate rate difference
        rate_difference = supplier_row['Fully Loaded Rate'] - benchmark_row['Benchmark Avg Rate']

        # Create a new row with the calculated values
        new_row = {
            'Supplier Type': supplier_row['Supplier ID'],  # Generate Supplier Type from Supplier ID
            'Supplier Department': supplier_row['Department'],
            'Supplier Role Title': supplier_row['Role Title'],
            'Supplier Role Description': supplier_row['Role Description'],
            'Benchmark Department': benchmark_row['Department'],
            'Benchmark Role Title': benchmark_row['Role Title'],
            'Benchmark Role Description': benchmark_row['Role Description'],
            'Benchmark Avg Rate': benchmark_row['Benchmark Avg Rate'],
            'Confidence Score': similarity_scores[i][idx],
            'Fully Loaded Rate': supplier_row['Fully Loaded Rate'],
            'Rate Difference': rate_difference,
            'Years of Experience': supplier_row['Years of Experience'],
            'Estimated Hours': supplier_row['Estimated Hours'],
            'Currency': supplier_row['Currency'],
            'Total Rate': supplier_row['Total Rate']
        }

        # Calculate Total Cost Difference and avoid negative values
        new_row['Total Cost Difference'] = max(rate_difference * supplier_row['Estimated Hours'], 0)

        # Assign threshold categories based on similarity scores
        if similarity_scores[i][idx] > 0.8:
            new_row['Threshold'] = 'High'
        elif 0.7 <= similarity_scores[i][idx] <= 0.8:
            new_row['Threshold'] = 'Medium'
        else:
            new_row['Threshold'] = 'Low'

        # Add the new row to the DataFrame
        all_supplier_results = all_supplier_results.append(new_row, ignore_index=True)


# Sort the DataFrame based on Confidence Score in descending order
all_supplier_results = all_supplier_results.sort_values('Confidence Score', ascending=False)

# Reset the index of the DataFrame
all_supplier_results = all_supplier_results.reset_index(drop=True)
# Feedback loop
while True:
    feedback = input("Please provide feedback on the similarity match score (Y/N): ")
    if feedback.upper() == "Y":
        # Get user feedback input
        feedback_department = input("Enter the feedback department: ")
        feedback_role_title = input("Enter the feedback role title: ")
        feedback_role_description = input("Enter the feedback role description: ")

        # Add feedback data to benchmark data
        benchmark_data = benchmark_data.append({
            'Department': preprocess_text(feedback_department),
            'Role Title': preprocess_text(feedback_role_title),
            'Role Description': preprocess_text(feedback_role_description),
            'Text': preprocess_text(feedback_department) + ' ' + preprocess_text(feedback_role_title) + ' ' + preprocess_text(feedback_role_description)
        }, ignore_index=True)

        # Retrain the model with the updated benchmark data
        benchmark_embeddings = model.encode(benchmark_data['Text'].tolist())

        # ... (Previous code remains the same)

        # Save the updated benchmark dataset to a CSV file
        benchmark_data.to_csv('updated_benchmark.csv', index=False)

    elif feedback.upper() == "N":
        print("Thank you for using the system.")
        break

    else:
        print("Invalid input. Please enter Y or N.")




# # Save the final supplier results to a CSV file
all_supplier_results.to_csv('output_supplier.csv', index=False)



