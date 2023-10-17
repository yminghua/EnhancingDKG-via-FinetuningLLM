from RDScore import *

def extract_and_rank(filename):
    # Load the Counter object from the .pkl file
    with open("./evaluation/corePath_Counter.pkl", "rb") as cfile:
        corePath_Counter = pickle.load(cfile)
        
    with open("./evaluation/subPath_Counter.pkl", "rb") as sfile:
        subPath_Counter = pickle.load(sfile)
    
    corePath_fmax = corePath_Counter.most_common(1)[0][1]
    subPath_fmax = subPath_Counter.most_common(1)[0][1]
    
    output = []
    with codecs.open(filename, "r", encoding="utf-8") as file:
        train_data = json.load(file)
        total_num = len(train_data) - 1
        
    for idx, item in enumerate(train_data):
        print(f"Processing {idx}/{total_num}...")
        output_data = {}
        sentence = item['target']
        output_data['pair'] = item['pair']
        output_data['relation_description'] = sentence
        doc = nlp(sentence)
        corePath = get_corePath(doc, item['pair'])
        if corePath == None:
            corePath = []
        output_data['Explicitness'] = cal_Explicitness(corePath, corePath_Counter, corePath_fmax)
        output_data['Significance'] = cal_Significance(doc, get_corePath_idx(doc, item['pair']), subPath_Counter, subPath_fmax)
        if (output_data['Explicitness'] + output_data['Significance']) == 0:
            output_data['RDScore'] = 0
        else:
            output_data['RDScore'] = (2*output_data['Explicitness']*output_data['Significance']) / (output_data['Explicitness'] + output_data['Significance'])
        output.append(output_data)
        
    sorted_output_desc = sorted(output, key=lambda x: x['RDScore'], reverse=True)
    
    with open(filename, 'w') as out_file:
        json.dump(sorted_output_desc, out_file)  
        

if __name__ == '__main__':
    
    data_folder = ''
    if len(sys.argv) == 2:
        data_folder = sys.argv[1] 
        file_list = ['train.json', 'dev.json', 'test.json']
        for file in file_list:
            extract_and_rank(data_folder + file)
    else:
        print("Missing target data folder name as argument!")
