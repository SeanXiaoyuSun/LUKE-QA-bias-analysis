 import json
 import sys
 def sample_metrics(record,file_path,suffix,luke_file,path1):
    import random
    data2=json.load(open(file_path,'rb'))
    idx=random.sample(range(len(data2)),record*2)
    samples_raw=[list(data2.keys())[x] for x in idx]
    filtered_sample=[]
    for n in range(record):
        if samples_raw[n] not in filtered_sample:
            filtered_sample.append(samples_raw[n])
            template=samples_raw[n].split('|')
            template[2],template[3]=template[3],template[2]
            conter_template='|'.join(template)

            filtered_sample.append(conter_template)
        if len(filtered_sample)==record:
            break
    assert(len(filtered_sample)==len(set(filtered_sample)))
    assert(len(filtered_sample)==record)

    _id=0
    data_sample={'data': [{'title':'sample_title', 'paragraphs':[]}],'version' : 1 }
    data_sample2={'data': [{'title':'sample_title', 'paragraphs':[]}],'version' : 1 }
    for d in list(data2.keys()):
        content=data2[d]
        inter_dic={'context':content['context'],'qas':[]}
        for q in list(content.keys())[1:]:
            inter_dic['qas'].append({'answers': [{'answer_start': '', 'text': content[q]['ans0']['text']},
               {'answer_start': '', 'text': content[q]['ans1']['text']}],
              'question': content[q]['question'],
              'id': d+'_'+str(_id)})
            _id+=1
        _id=0
        if d in filtered_sample:
            data_sample['data'][0]['paragraphs'].append(inter_dic)
        else:
            data_sample2['data'][0]['paragraphs'].append(inter_dic)
    assert((len(data_sample2['data'][0]['paragraphs'])+len(data_sample['data'][0]['paragraphs']))==len(list(data2.keys())))
    print((len(data_sample2['data'][0]['paragraphs']),len(data_sample['data'][0]['paragraphs'])),len(list(data2.keys())))

    
    with open(f"{luke_file}/data/unqover_{suffix}.json", "w") as outfile: 
        json.dump(data_sample, outfile)        
    with open(f"{path1}/unqover_{suffix}_remian.json", "w") as outfile: 
        json.dump(data_sample2, outfile) 
if __name__ == "__main__":
    record = sys.argv[1]
    file_path = sys.argv[2]
    suffix = sys.argv[3]
    luke_file = sys.argv[4]
    path1 = sys.argv[5]
	sample_metrics(record,file_path,suffix,luke_file,path1)