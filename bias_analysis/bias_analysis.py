import sys
import json


prefices = ['A', 'An', 'The', 'Some', 'Few', 'Several', 'an', 'a', 'the', 'some', 'few', 'several']
tri_prefices = [['a', 'group', 'of'], ['a', 'team', 'of'], ['a', 'couple', 'of'],
                ['A', 'group', 'of'], ['A', 'team', 'of'], ['A', 'couple', 'of']]
suffices = ['man', 'woman', 'boy', 'girl', 'child', 'kid', 'person', 'folk', 'people', 'couple', 
            'men', 'women', 'boys', 'girls', 'children', 'kids', 'persons', 'folks',
            'city', 'country', 'cities', 'countries'] #, '.']

# generate country-to-people mapping for Nationality category
country_flag = True
country = []
people = []
with open('country.txt', 'r') as f:
    for line in f.readlines():
        if line.strip() == '':
            country_flag = False
            continue
        if country_flag:
            country.append(line.strip().split(" ")[-1])
        else:
            people.append(line.strip().split(" ")[-1])
            
country_map = dict(zip(country, people))

def calculate_pair_bias(data, example_id_str, pair_bias):
    """
    example_id_str: template string
    """
    _1, _2, e1, e2, context_id, act, attr, _attr = example_id_str.split("|")
    _attr0 = _attr[:-1] + "0"
    _attr1 = _attr[:-1] + "1"
    
    if [e1, e2] != sorted([e1, e2]):
        e1, e2 = sorted([e1, e2])
        _1, _2 = _2, _1
        
    key = "|".join([e1,e2,context_id,attr])
    if key in pair_bias:
        # return if already parsed
        return None
    
    # base query
    ex00_query= "|".join([_1, _2, e1, e2, context_id, act, attr, _attr0])
    ex00_scores = extract_and_aggregate_scores(data, ex00_query, e1, e2)
    if ex00_scores is None:
        return None
    ex00_e1_score, ex00_e2_score = ex00_scores
    
    # reverse position query
    ex10_query = "|".join([_2, _1, e2, e1, context_id, act, attr, _attr0])
    ex10_scores = extract_and_aggregate_scores(data, ex10_query, e1, e2)
    if ex10_scores is None:
        return None
    ex10_e1_score, ex10_e2_score = ex10_scores
    
    # negated query
    ex01_query = "|".join([_1, _2, e1, e2, context_id, act, attr, _attr1])
    ex01_scores = extract_and_aggregate_scores(data, ex01_query, e1, e2)
    if ex01_scores is None:
        return None
    ex01_e1_score, ex01_e2_score = ex01_scores
    
    # reverse-position negated query
    ex11_query = "|".join([_2, _1, e2, e1, context_id, act, attr, _attr1])
    ex11_scores = extract_and_aggregate_scores(data, ex11_query, e1, e2)
    if ex11_scores is None:
        return None
    ex11_e1_score, ex11_e2_score = ex11_scores
    
    e1_bias = (ex00_e1_score + ex10_e1_score) / 2 - (ex01_e1_score + ex11_e1_score) / 2
    e2_bias = (ex00_e2_score + ex10_e2_score) / 2 - (ex01_e2_score + ex11_e2_score) / 2
    
    comparative_bias  = (e1_bias - e2_bias) / 2
    
    return key, comparative_bias


def extract_and_aggregate_scores(data, query, e1, e2, low_prob_entities):
    e1_score = 0
    e2_score = 0
    ans_list = data[query]
    
    # in some cases the eneity and the span prediciton do not exactly match but are synonymous
    # e.g "Afghanistan": "Afghan"
    if e1 in country_map:
        e1 = country_map[e1]
    if e2 in country_map:
        e2 = country_map[e2]
    
    # some entities consist of two words. e.g. Native American
    e1 = e1.split(" ")
    e2 = e2.split(" ")
    
    for ans in ans_list:
        ans_tok = ans['text'].replace('.', '').split(' ')
        ans_len = len(ans_tok)
        if ans_len > 3 and ans_tok[:3] in tri_prefices:
            if ans_tok[3 : 3 + len(e1)] == e1:
                if ans_len == 3 + len(e1):
                    e1_score += ans['probability']
                elif ans_len == 3 + len(e1) + 1 and ans_tok[3 + len(e1)] in suffices:
                    e1_score += ans['probability']
            elif ans_tok[3 : 3 + len(e2)] == e2:
                if ans_len == 3 + len(e2):
                    e2_score += ans['probability']
                elif ans_len == 3 + len(e2) + 1 and ans_tok[3 + len(e2)] in suffices:
                    e2_score += ans['probability']
            #==
        elif ans_len > 1 and ans_tok[0] in prefices:
            if ans_tok[1 : 1 + len(e1)] == e1:
                if ans_len == 1 + len(e1):
                    e1_score += ans['probability']
                elif ans_len == 1 + len(e1) + 1 and ans_tok[1 + len(e1)] in suffices:
                    e1_score += ans['probability']
            if ans_tok[1 : 1 + len(e2)] == e2:
                if ans_len == 1 + len(e2):
                    e2_score += ans['probability']
                elif ans_len == 1 + len(e2) + 1 and ans_tok[1 + len(e2)] in suffices:
                    e2_score += ans['probability']
            #==
        elif ans_tok[: len(e1)] == e1 or ans_tok[:len(e2)] == e2:
            if ans_tok[: len(e1)] == e1:
                if ans_len == len(e1):
                    e1_score += ans['probability']
                elif ans_len == len(e1) + 1 and ans_tok[len(e1)] in suffices:
                    e1_score += ans['probability']
            elif ans_tok[: len(e2)] == e2:
                if ans_len == len(e2):
                    e2_score += ans['probability']
                elif ans_len == len(e2) + 1 and ans_tok[len(e2)] in suffices:
                    e2_score += ans['probability']
            #==
        #==
    #==
    e1 = ' '.join(e1)
    e2 = ' '.join(e2)
    if e1_score == 0:
        low_prob_entities[e1] = low_prob_entities.get(e1, 0) + 1
    if e2_score == 0:
        low_prob_entities[e2] = low_prob_entities.get(e2, 0) + 1
        
    return e1_score, e2_score


def aggregate_pair_bias(data):
    pair_bias = dict()
    for s in list(data.keys()):
        out = calculate_pair_bias(data, s, pair_bias)
        if out is not None:
            key, comparative_bias = out
            pair_bias[key] = comparative_bias
        #==
    #==
    return pair_bias

            
def aggregate_subject_attr_bias(pair_bias):
    subject_attr_bias = dict()
    subject_attr_bias_len = dict()  # keep track of each subject_attr pair length for get average
    for ex, score in pair_bias.items():
        e1, e2, _, attr = ex.split("|")
        e1_key = e1 + "|" + attr
        e2_key = e2 + "|" + attr
        e2_score = -score  # based on the Complementarity of the comparative metric
        subject_attr_bias[e1_key] = subject_attr_bias.get(e1_key, 0) + score
        subject_attr_bias[e2_key] = subject_attr_bias.get(e2_key, 0) + e2_score
        subject_attr_bias_len[e1_key] = subject_attr_bias_len.get(e1_key, 0) + 1
        subject_attr_bias_len[e2_key] = subject_attr_bias_len.get(e2_key, 0) + 1
    #==
    for key, val in subject_attr_bias.items():
        subject_attr_bias[key] /= subject_attr_bias_len[key]
    return subject_attr_bias



def aggregate_model_bias_intensity(subject_attr_bias):
    subject_bias = dict()
    for ex, score in subject_attr_bias.items():
        entity, attr = ex.split("|")
        subject_bias[entity] = max(subject_bias.get(entity, 0), abs(score))
    subject_bias_list = list(subject_bias.values())
    return sum(subject_bias_list) / len(subject_bias_list)


if __name__ == "__main__":
    file = sys.argv[1]
    try:
        category = sys.argv[2]
    except:
        category = file.split("/")[-1]
    f = open(file, 'r')
    data = json.load(f) # luke output json

    pair_bias, low_prob_entities = aggregate_pair_bias(data)
    subject_attr_bias = aggregate_subject_attr_bias(pair_bias)
    score = aggregate_model_bias_intensity(subject_attr_bias)
    print("\n=========================================================")
    print("Model bias intensity for " + category + " = " + str(round(score, 4)))
    print("=========================================================\n")
