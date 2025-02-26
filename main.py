
import pandas as pd
from transformers import pipeline

def classify_column_zero_shot(column_data, candidate_labels):
    """Classifies column data using zero-shot classification."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    column_string = " ".join(map(str, column_data))
    result = classifier(column_string, candidate_labels)
    return result["labels"][0]

def suggest_headers_zero_shot(csv_file):
    """Suggests headers using zero-shot classification from a CSV."""
    try:
        df = pd.read_csv(csv_file, header=None, nrows=5) #read only the first 5 rows
        suggested_headers = []

        potential_headers = ["code", "title", "category","currency", "email address", "quantity", "name", "date", "id", "value", "quantity", "description", "address", "email", "phone number", "product", "price", "city", "country", "state", "zip code", "customer", "order", "status", "link"]

        for column in df.columns:
            column_data = df[column].astype(str).tolist()
            predicted_header = classify_column_zero_shot(column_data, potential_headers)
            suggested_headers.append(predicted_header)

        return suggested_headers

    except FileNotFoundError:
        return "Error: CSV file not found."
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage (replace with your CSV file):
csv_file = "DemandPlan_v1.csv"


suggested_headers = suggest_headers_zero_shot(csv_file)
print(suggested_headers)