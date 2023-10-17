import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import codecs
import nltk
# nltk.download('punkt')  # Download the necessary data for sentence tokenization

output_merged_dir = "results/llama2_finetune"

### Inference
# Specify device (single GPU)
"""
export CUDA_VISIBLE_DEVICES=0
"""
n_gpus = sum(1 for _ in range(torch.cuda.device_count()) if torch.cuda.is_available())
print(f"Available GPU: {n_gpus}")
device = torch.device("cuda:0")

# Load the model
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map=device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_merged_dir, device_map=device)

def get_first_sentence(p):
        sentences = nltk.sent_tokenize(p)
        if sentences:
            return sentences[0].strip()
        return None


def generate_relation_description(pair):

    prompt_templete = "Generate explicit, concise and correct relation description between a pair of entities in one simple short sentence. The generated sentence should only contain a 'argument-predicate-argument' structure connecting the two entities. You can refer to the wikipedia summary of each concept entity for some information. Only response the final description sentence. ### Input: "

    # Tokenize input text
    if isinstance(pair, list):
        prompt = prompt_templete + str(pair)
    elif isinstance(pair, str):
        prompt = prompt_templete + pair
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get answer
    # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    # Decode output & print it
    decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split the text into lines
    lines = decode_output.split("###")

    # Find the line containing "Response:"
    response_line = [line.strip() for line in lines if "Response:" in line]
    if len(response_line) == 0:
        return None

    # Extract the words after "Response:"
    response_words = response_line[0].split("Response:")[-1].strip()

    return get_first_sentence(response_words)



if __name__ == '__main__':
    with codecs.open("./test_pairs.json", "r", encoding="utf-8") as pair_file:
        test_pair_list = json.load(pair_file)

    total_num = len(test_pair_list)
    success_list = []
    bug_list = []
    for idx, concept_pair in enumerate(test_pair_list):
        print(f"Processing {idx}/{total_num-1}...")
        generated_description = generate_relation_description(concept_pair)
        if generated_description != None:
            success_list.append({"pair": concept_pair, "relation_description": generated_description})
        else:
            bug_list.append(concept_pair)
                
    with open('./f-llama2_output.json', 'w') as out_file:
        json.dump(success_list, out_file)
    if len(bug_list) > 0:
        with open('./f-llama2-bug_list.json', 'w') as ob_file:
            json.dump(bug_list, ob_file)
    