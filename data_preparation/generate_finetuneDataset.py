import json
import codecs
import csv
import sys

def generate_finetuneData(filename, threshold=0.75):

    with codecs.open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    instruction_based_data = []
    for item in data:
        if item['RDScore'] < threshold:
            break
        else:
            insturction = "Generate explicit, concise and correct relation description between a pair of entities in one simple short sentence. The generated sentence should only contain a 'argument-predicate-argument' structure connecting the two entities. You can refer to the wikipedia summary of each concept entity for some information. Only response the final description sentence."
            input = str(item['pair'])
            output = item['relation_description']
            text = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: " + insturction + " ### Input: " + input + " ### Response: " + output
            instruction_based_data.append({"instruction": insturction, "input": input, "output": output, "text": text})
        
    with open(filename.split('.')[0]+'_slice.csv', mode='w', newline='', encoding="utf-8") as outfile:
        fieldnames = ["instruction", "input", "output", "text"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write each row of data
        total_samples = len(instruction_based_data)
        print(f"Retrieve {total_samples} samples in {filename}")
        for row in instruction_based_data:
            writer.writerow(row)
        
        
if __name__ == '__main__':
    
    if len(sys.argv) <= 1:
        generate_finetuneData('train.json')
        generate_finetuneData('dev.json')
    else:
        threshold = float(sys.argv[1])
        generate_finetuneData('train.json', threshold)
        generate_finetuneData('dev.json', threshold)
    