from datasets import load_dataset
import csv

# Load the dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split='train_gen')

assistant_messages = []

# Iterate through each item in the dataset
for item in dataset:
    for message in item['messages']:
        if message['role'] == 'assistant':
            print(message['content'])
            assistant_messages.append(message['content'])

# Define the CSV file path
csv_file_path = 'Semantic_Similarity/assistant_messages.csv'

# Save the assistant messages to a CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['content'])  # Writing header
    for content in assistant_messages:
        writer.writerow([content])

print(f"CSV file has been saved to {csv_file_path}")


