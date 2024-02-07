import torch
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the label_to_id mapping

file_path = './Classification_Model/label_to_id.json'

with open(file_path, 'r') as f:
    label_to_id = json.load(f)
id_to_label = {id: label for label, id in label_to_id.items()}

# Load model and tokenizer
model_path = '/Users/tenboker/Desktop/Code Shtuff/NLPschtuff/WinterTerm2024-NLPModel/Classification_Model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    # Encode the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities (softmax)
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Get the predicted class ID and label
    predicted_class_id = probabilities.argmax().item()
    predicted_class_label = id_to_label[predicted_class_id]  # Use the id_to_label mapping

    return predicted_class_label

if __name__ == "__main__":
    # Test the function
    test_text = input("Enter test text:")
    prediction = predict(test_text)
    print(f"Predicted class for the input text: {prediction}")
