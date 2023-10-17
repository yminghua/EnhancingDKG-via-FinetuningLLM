import spacy
import json
import codecs
import os
import sys
import pickle
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine


# Load the English language model
nlp = spacy.load("en_core_web_lg")

def get_root_token(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token
        
        
# for both Explicitness and Significance chosen by argument 'purpose', see comments below
def find_dependency_path(doc, start_token, end_token, purpose:str):
    
    path_to_startT = []
    path_to_endT = []
    
    # Initialize a queue for BFS and a dictionary to keep track of visited tokens
    root_token = get_root_token(doc)
    queue = [(root_token, [])]
    visited = {root_token: True}

    # Perform BFS
    while queue:
        current_token, path = queue.pop(0)
        
        # If we reached the target token, return the path
        if current_token.i in start_token:
            if purpose == 'Explicitness':
                path_to_startT.append(["i_" + dep for dep in path])    # for Explicitness
            elif purpose == 'Significance':
                path_to_startT.append(path)   # for Significance
        elif current_token.i in end_token:
            path_to_endT.append(path)
        
        # Explore the children of the current token
        for child in current_token.children:
            if child not in visited:
                visited[child] = True
                if purpose == 'Explicitness':
                    queue.append((child, path + [child.dep_]))    # for Explicitness
                elif purpose == 'Significance':
                    queue.append((child, path + [child]))   # for Significance
    
    if len(path_to_startT) == 0 or len(path_to_endT) == 0:
        return None, None
    
    return min(path_to_startT, key=len), min(path_to_endT, key=len)


def cal_token_similarity(token1, token2):
    vector1 = token1.vector
    vector2 = token2.vector
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0
    
    return 1 - cosine(vector1, vector2)


def match_entity(entity, doc):
    
        idx_list = []
        similarity_threshold = 0.8
        
        if len(entity) > 1:
            for i, token in enumerate(doc):
                current_token = token.text.lower()
                check_next = True
                if current_token == entity[0].text.lower():
                    for j in range(1, len(entity)):
                        if (i+j) < len(doc):
                            check_next = check_next and (doc[i+j].text.lower() == entity[j].text.lower())
                    if check_next:
                        upper_bound = i+len(entity)
                        if upper_bound > len(doc):
                            upper_bound = len(doc)
                        for ii in range(i, upper_bound):
                            idx_list.append(ii)
            if len(idx_list) == 0:
                for i, token in enumerate(doc):
                    current_token = token.text.lower()
                    if current_token == entity[0].text.lower() or cal_token_similarity(token, entity[0]) > similarity_threshold:
                        idx_list.append(i)
                        for j in range(1, len(entity)):
                            if (i+j) < len(doc):
                                if doc[i+j].text.lower() == entity[j].text.lower() or cal_token_similarity(doc[i+j], entity[j]) > similarity_threshold:
                                    idx_list.append(i+j)
        else:
            for i, token in enumerate(doc):
                current_token = token.text.lower()
                if current_token == entity[0].text.lower():
                    idx_list.append(i)
            if len(idx_list) == 0:
                for i, token in enumerate(doc):
                    if cal_token_similarity(token, entity[0]) > similarity_threshold:
                        idx_list.append(i)
        
        return idx_list


def find_concept_pair_idx(doc, concept_pair):
    entity1 = nlp(concept_pair[0])
    entity2 = nlp(concept_pair[1])
    
    entity1_idx = match_entity(entity1, doc)
    entity2_idx = match_entity(entity2, doc)
    
    if len(entity1_idx) == 0 or len(entity2_idx) == 0:
        return None, None
    
    if entity1_idx[0] <= entity2_idx[0]:
        start_token = entity1_idx
        end_token = entity2_idx
    else:
        start_token = entity2_idx
        end_token = entity1_idx
        
    return start_token, end_token


def get_corePath(doc, concept_pair):
    
    start_token, end_token = find_concept_pair_idx(doc, concept_pair)
    
    if start_token == None or end_token == None:
        # print(f"{concept_pair} can't match index!")
        return None
    
    path_to_startT, path_to_endT = find_dependency_path(doc, start_token, end_token, 'Explicitness')
    if path_to_startT == None or path_to_endT == None:
        return None
    
    corePath = path_to_startT[::-1] + path_to_endT
    corePath = [dep for dep in corePath if dep not in ['conj', 'appos', 'i_conj', 'i_appos']]
    if len(corePath) == 0:
        return None
    
    consecutive_prep = []
    for idx, dep in enumerate(corePath):
        if dep == 'prep':
            if idx < len(corePath) - 1:
                if corePath[idx+1] == 'prep':
                    consecutive_prep.append(idx+1)
                
    corePath = [dep for i, dep in enumerate(corePath) if i not in consecutive_prep]
    
    if corePath[0] not in ['nsubj', 'i_nsubj', 'nsubjpass', 'i_nsubjpass']:
        # print(f"{corePath} is not valid for {concept_pair}")
        # print(f"start_token: {start_token}, end_token: {end_token}")
        # for token in doc:
        #     print(f"{token.text} <--{token.dep_}-- {token.head}")
        # print("\n")
        return None
    
    return corePath


def get_corePath_idx(doc, concept_pair):
    
    start_token, end_token = find_concept_pair_idx(doc, concept_pair)
    
    if start_token == None or end_token == None:
        return None
    
    path_to_startT, path_to_endT = find_dependency_path(doc, start_token, end_token, 'Significance')
    if path_to_startT == None or path_to_endT == None:
        return None
    
    corePath = path_to_startT[::-1] + [get_root_token(doc)] + path_to_endT
    if len(corePath) == 0:
        return None
    
    corePath_idx = [token.i for token in corePath]
    
    return corePath_idx


def find_subPath_pattern(doc, corePath_idx, target_idx):
    
    path_to_corePath = []
    path_to_target = []
    
    # Initialize a queue for BFS and a dictionary to keep track of visited tokens
    root_token = get_root_token(doc)
    queue = [(root_token, [root_token.i])]
    visited = {root_token: True}

    # Perform BFS
    while queue:
        current_token, path = queue.pop(0)
        
        # If we reached the target token, return the path
        if current_token.i in corePath_idx:
            path_to_corePath.append(path)
        elif current_token.i == target_idx:
            path_to_target = path
        
        # Explore the children of the current token
        for child in current_token.children:
            if child not in visited:
                visited[child] = True
                queue.append((child, path + [child.i]))
    
    if len(path_to_corePath) == 0 or len(path_to_target) == 0:
        return None, None
    
    subPath_list = []
    for c_path in path_to_corePath:
        if c_path[-1] <= path_to_target[-1]:
            start_path = c_path
            end_path = path_to_target
            connect_dir = "s"
        else:
            start_path = path_to_target
            end_path = c_path
            connect_dir = "e"
            
        check_list = list(zip(start_path, end_path))
        for idx, item in enumerate(check_list):
            if item[0] == item[1]:
                remove_idx = idx

        if remove_idx:
            start_path = start_path[(remove_idx+1):]
            end_path = end_path[(remove_idx+1):]
                
        start_path = ["i_" + doc[i].dep_ for i in start_path]
        end_path = [doc[i].dep_ for i in end_path]
        subPath_list.append(start_path[::-1] + end_path + [connect_dir])
        
    shortest_path = min(subPath_list, key=len)
    if shortest_path[-1] == "s":
        connect_dep = shortest_path[0]
    elif shortest_path[-1] == "e":
        connect_dep = shortest_path[-2]
        
    return shortest_path[:-1], connect_dep


def get_subPath(doc, corePath_idx):
    
    subPath_list = []
    rest_token_idx = [i for i in range(len(doc)-1) if i not in corePath_idx]
    for ri in rest_token_idx:
        subPath, _ = find_subPath_pattern(doc, corePath_idx, ri)
        if subPath != None:
            subPath = [dep for dep in subPath if dep not in ['conj', 'appos', 'compound', 'i_conj', 'i_appos', 'i_compound']]
            if len(subPath) > 0:
                subPath_list.append(tuple(subPath))
    
    return subPath_list


def cal_Explicitness(corePath, corePath_Counter, fmax):
    fp = corePath_Counter[tuple(corePath)]
    if fp == 0:
        fp == 0.5
        
    return np.log(fp + 1) / np.log(fmax + 1)


def cal_Significance(doc, corePath_idx, subPath_Counter, fmax):
    
    if corePath_idx == None:
        return 0
    
    modifying_dependency = ['acl', 'advcl', 'advmod', 'amod', 'det', 'mark', 'meta', 'neg', 'nn', 'nmod', 'npmod', 'nummod', 'poss', 'prep', 'quantmod', 'relcl', 'appos', 'aux', 'auxpass', 'compound', 'cop', 'ccomp', 'xcomp', 'expl', 'punct', 'nsubj', 'csubj', 'csubjpass', 'dobj', 'iobj', 'obj', 'pobj', 'i_acl', 'i_advcl', 'i_advmod', 'i_amod', 'i_det', 'i_mark', 'i_meta', 'i_neg', 'i_nn', 'i_nmod', 'i_npmod', 'i_nummod', 'i_poss', 'i_prep', 'i_quantmod', 'i_relcl', 'i_appos', 'i_aux', 'i_auxpass', 'i_compound', 'i_cop', 'i_ccomp', 'i_xcomp', 'i_expl', 'i_punct', 'i_nsubj', 'i_csubj', 'i_csubjpass', 'i_dobj', 'i_iobj', 'i_obj', 'i_pobj']
    
    Sigscore = 0
    total_num = len(doc) - 1
    
    for i in range(len(doc)-1):
        if i in corePath_idx:
            Sigscore += 1
        else:
            subPath, connect_dep = find_subPath_pattern(doc, corePath_idx, i)
            if subPath != None and connect_dep != None:
                # if connect_dep in modifying_dependency:
                if len(set(subPath).intersection(set(modifying_dependency))) > 0:
                    fp = subPath_Counter[tuple(subPath)]
                    if fp == 0:
                        fp = 0.5
                    Sigscore += np.log(fp + 1) / np.log(fmax + 1)
            else:
                total_num -= 1
                
    return Sigscore / total_num
    

def process_file_calScore(filename):
    
    # Load the Counter object from the .pkl file
    with open("./evaluation/corePath_Counter.pkl", "rb") as cfile:
        corePath_Counter = pickle.load(cfile)
        
    with open("./evaluation/subPath_Counter.pkl", "rb") as sfile:
        subPath_Counter = pickle.load(sfile)
    
    corePath_fmax = corePath_Counter.most_common(1)[0][1]
    subPath_fmax = subPath_Counter.most_common(1)[0][1]
    
    output = []
    with open(filename, "r", encoding="utf-8") as file:
        description_list = json.load(file)
        
    for item in description_list:
        sentence = item['relation_description']
        doc = nlp(sentence)
        corePath = get_corePath(doc, item['pair'])
        if corePath == None:
            corePath = []
        item['Explicitness'] = cal_Explicitness(corePath, corePath_Counter, corePath_fmax)
        item['Significance'] = cal_Significance(doc, get_corePath_idx(doc, item['pair']), subPath_Counter, subPath_fmax)
        if (item['Explicitness'] + item['Significance']) == 0:
            item['RDScore'] = 0
        else:
            item['RDScore'] = (2*item['Explicitness']*item['Significance']) / (item['Explicitness'] + item['Significance'])
        output.append(item)
        
    with open(filename, 'w') as out_file:
        json.dump(output, out_file)
        
        
def process_file_avgScore(filename):
    
    base_name, _ = os.path.splitext(filename)
    
    with open(filename, "r", encoding="utf-8") as file:
        description_list = json.load(file)
        
    total_num = len(description_list)
        
    def cal_avg_score(description_list, total_num, score_type):
        score = 0
        for item in description_list:
            score += item[score_type]
            if item[score_type] == 0:
                total_num -= 1
                
        return score / total_num
        
    print(f"{base_name}: ")
    print(f"Avg_ExpScore = {cal_avg_score(description_list, total_num, 'Explicitness')}; Avg_SigScore = {cal_avg_score(description_list, total_num, 'Significance')}; Avg_RDScore = {cal_avg_score(description_list, total_num, 'RDScore')}")
    print("\n")
    


if __name__ == '__main__': 
    
    def generate_subPath_Counter():
        
        # file_structure: [{'concept_pair': [], 'relation_description': ""}]
        with codecs.open("./evaluation/wiki_corpus.json", "r", encoding="utf-8") as file:
            dataset = json.load(file)
            
        total_num = len(dataset)
        wikiCorpus_subPath = []
            
        for idx, item in enumerate(dataset):
            print(f"Processing {idx}/{total_num-1}...")
            sentence = item['relation_description']
            doc = nlp(sentence)
            corePath_idx = get_corePath_idx(doc, item['concept_pair'])
            if corePath_idx != None:
                wikiCorpus_subPath += get_subPath(doc, corePath_idx)

        subPath_Counter = Counter(wikiCorpus_subPath)
        
        with open("./evaluation/subPath_Counter.pkl", "wb") as file:
            pickle.dump(subPath_Counter, file)
        
    
    def generate_corePath_Counter():
        
        # file_structure: [{'concept_pair': [], 'relation_description': ""}]
        with codecs.open("./evaluation/wiki_corpus.json", "r", encoding="utf-8") as file:
            dataset = json.load(file)
            
        total_num = len(dataset)
        invalid_cnt = 0
            
        wikiCorpus_corePath = []
            
        for idx, item in enumerate(dataset):
            print(f"Processing {idx}/{total_num-1}...")
            sentence = item['relation_description']
            doc = nlp(sentence)
            corePath = get_corePath(doc, item['concept_pair'])
            if corePath != None:
                wikiCorpus_corePath.append(tuple(corePath))
            else:
                invalid_cnt += 1

        corePath_Counter = Counter(wikiCorpus_corePath)
        
        with open("./evaluation/corePath_Counter.pkl", "wb") as file:
            pickle.dump(corePath_Counter, file)
        
        print(f"Total_num: {total_num}, Invalid_num: {invalid_cnt}")
        
    
    if len(sys.argv) <= 1:
        print("Missing file names as arguments!")
    elif sys.argv[1] == "generate_counters":
        generate_corePath_Counter()
        generate_subPath_Counter()
    else:
        file_list = sys.argv[1:]
        for file in file_list:
            if file == "baseline":
                process_file_avgScore('./data_preparation/test.json')
            else:
                process_file_calScore(file)
                process_file_avgScore(file)
