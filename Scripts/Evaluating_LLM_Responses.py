import time,random,requests,pandas as pd,numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk import word_tokenize
# from nltk.translate import meteor_score as ms
import evaluate,csv

def save_response_to_file(response):
    with open(output_file, "a", encoding='utf-8') as file:
        file.write(response + "\n\n")
        
def get_random_line_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        return random.choice(lines).strip()

def create_world_background_prompts_dict(file_path):
    world_background_prompts = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split("|", 1)  # This splits at the first '|' only
                world_background = parts[0].strip()

                # Now, split the second part to separate the prompt from the length
                background_prompt, length = parts[1].rsplit("|", 1)

                # Stripping potential leading and trailing spaces
                world_background = world_background.strip()
                background_prompt = background_prompt.strip()
                length = length.strip()
                
                world_background_prompts[world_background] = {
                    "prompt": background_prompt,
                    "length": length
                }
    return world_background_prompts

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

world_background_file = "World_Details\world_backgrounds.txt"
world_background_prompts_file = "World_Details\world_background_prompts.txt"
character_details_file = "World_Details\character_details.txt"
world_background_prompts = create_world_background_prompts_dict(world_background_prompts_file)
API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# Load Cached Reference Embeddings
reference_embeddings = np.load('Semantic_Similarity/reference_embeddings.npy')

# Load the dataset to match the best reference text
df = pd.read_csv('Semantic_Similarity/assistant_messages.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

output_file = "Testing_generated_responses.txt"

while(1):
    #Pre-step: Loading random world, character and question
    # world_background = get_random_line_from_file(world_background_file)
    world_background = get_random_line_from_file(world_background_file).split("|",1)
    world_background_prompt = world_background[0].strip()
    world_background_length = world_background[1].strip()
    # character_details = get_random_line_from_file(character_details_file)
    character_details = get_random_line_from_file(character_details_file).split("|",1)
    character_details_prompt = character_details[0].strip()
    character_details_length = character_details[1].strip()
    user_content = world_background_prompts.get(world_background_prompt)
    user_prompt = user_content["prompt"]
    # prompt_length = user_content["length"]
    
    #------------------------------------------------------------------------------------------------
    #Step 1: Generating a response from the LLM
    formatted_prompt = f"<|system|>You will respond the given prompt in the first-person and within your character's context.The setting of the environment is: {world_background_prompt}. You are: {character_details_prompt}.</s>\n<|user|>{user_prompt}</s><|assistant|>"
    
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    data = query({
    "inputs":formatted_prompt,
    "parameters": {
        "return_full_text": False,
        "max_new_tokens": 500
        }
    })
    generated_response = (data[0]["generated_text"])
    # print(generated_response)
    
    #---------------------------------------------------------------------------------------------
    #Step 2: Find 3 closest matching reference from pre-loaded embeddings

    candidate_embedding = model.encode([generated_response])  
    
    similarities = cosine_similarity(candidate_embedding, reference_embeddings).flatten()
    
    top_indices = np.argsort(similarities)[-3:]
    
    references_list = df['content'].tolist()
    top_references = [references_list[i] for i in top_indices]
    
    top_references.reverse()
    
    # print(top_references)
    
    #--------------------------------------------------------------------------------------------
    #Step 3: Get Metrics from predictions and references
    
    predictions = [generated_response]
    references = [top_references]
    
    bleu = evaluate.load('bleu')
    bleu_results = bleu.compute(predictions=predictions, references=references)
    # print(bleu_results)
    
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)
    # print(rouge_results)
    
    meteor = evaluate.load('meteor')
    m_results = meteor.compute(predictions=predictions,references=references) 
    # print(m_results)
    
    bertscore = evaluate.load('bertscore')
    bertscore_results = bertscore.compute(predictions=predictions, references=references,model_type="distilbert-base-uncased")
    # print(bertscore_results)
    
    #--------------------------------------------------------------------------------------------
    #Step 4: Saving Results
    with open('data.csv','a',newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([formatted_prompt,world_background_length,character_details_length,generated_response,bleu_results.get("bleu"),bleu_results.get("precisions"),rouge_results.get("rouge1"),rouge_results.get("rouge2"),rouge_results.get("rougeL"),rouge_results.get("rougeLsum"),m_results.get("meteor"),bertscore_results])
    
    #save_response_to_file(f"Generated Response: {generated_response}\nBest References: {top_references}\nPrompt Length: {prompt_length}\nBLEU Results: {bleu_results}\nROUGE Results: {rouge_results}\nMETEOR Results: {m_results}")
    print("Results Saved")
    time.sleep(10)
    