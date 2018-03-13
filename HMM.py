import sys
import codecs
import numpy as np

resultList = []


def counting_probabilities(file_path):  # returns unigram and bigram counts for probability matrix
    sentence_counter = 0
    previous_tag = None
    previouse_word_language = None
    tag_count_dictionary = {}
    spa_tag_count={}
    eng_tag_count={}
    transition_count = {}
    spa_transition_count={}
    eng_transition_count={}
    emission_count = {}
    spanish_emission_count={}
    english_emission_count={}
    word_counting = {}
    spa_word_counting ={}
    eng_word_counting ={}
    for line in open(file_path,'r',encoding='UTF-8'):
        list_line = []
        for col in line.split():
            list_line.append(col)
        if len(list_line) == 0:  # sentence ended computing transition from tag to end state
            if tuple(['END', previous_tag]) in transition_count:
                transition_count[tuple(['END', previous_tag])] += 1
            else:
                transition_count[tuple(['END', previous_tag])] = 1
            if previouse_word_language in ['spa','eng&spa','UNK']:
                if tuple(['END', previous_tag]) in spa_transition_count:
                    spa_transition_count[tuple(['END', previous_tag])]+=1
                else:
                    spa_transition_count[tuple(['END', previous_tag])]=1
            if previouse_word_language in ['eng','eng&spa','UNK']:
                if tuple(['END', previous_tag]) in eng_transition_count:
                    eng_transition_count[tuple(['END', previous_tag])]+=1
                else:
                    eng_transition_count[tuple(['END', previous_tag])]=1
            previous_tag = None  # for computing start to tag
            previouse_word_language=None
        else:
            #for smoothing
            if list_line[0] in word_counting:
                word_counting[list_line[0]]+=1
            else:
                word_counting[list_line[0]]=1
            if list_line[1] in ['spa','eng&spa','UNK']:
                if list_line[0] in spa_word_counting:
                    spa_word_counting[list_line[0]]+=1
                else:
                    spa_word_counting[list_line[0]]=1
            if list_line[1] in ['eng','eng&spa','UNK']:
                if list_line[0] in eng_word_counting:
                    eng_word_counting[list_line[0]]+=1
                else:
                    eng_word_counting[list_line[0]]=1

            if list_line[2] in tag_count_dictionary:  # counting tag occurrences
                tag_count_dictionary[list_line[2]] += 1  # counting delimiter for emission
            else:
                tag_count_dictionary[list_line[2]] = 1
            if list_line[1] in ['spa','eng&spa','UNK']:
                if list_line[2] in spa_tag_count:
                    spa_tag_count[list_line[2]]+=1
                else:
                    spa_tag_count[list_line[2]]=1
            if list_line[1] in ['eng','eng&spa','UNK']:
                if list_line[2] in eng_tag_count:
                    eng_tag_count[list_line[2]]+=1
                else:
                    eng_tag_count[list_line[2]]=1
            if previous_tag == None:  # computing probability from START to tag
                sentence_counter += 1
                if tuple([list_line[2], 'START']) in transition_count:  # increasing counter
                    transition_count[tuple([list_line[2], 'START'])] += 1
                else:  # creating new one
                    transition_count[tuple([list_line[2], 'START'])] = 1
                if list_line[1] in ['spa','eng&spa','UNK']:
                    if list_line[2] in spa_transition_count:
                        spa_transition_count[tuple([list_line[2], 'START'])]+=1
                    else:
                        spa_transition_count[tuple([list_line[2], 'START'])]=1
                if list_line[1] in ['eng','eng&spa','UNK']:
                    if list_line[2] in eng_transition_count:
                        eng_transition_count[tuple([list_line[2], 'START'])]+=1
                    else:
                        eng_transition_count[tuple([list_line[2], 'START'])]=1
            else:  # we are in the middle of sentence
                if tuple([list_line[2], previous_tag]) in transition_count:
                    transition_count[tuple([list_line[2], previous_tag])] += 1
                else:
                    transition_count[tuple([list_line[2], previous_tag])] = 1
                if list_line[1] in ['spa','eng&spa','UNK']:
                    if list_line[2] in spa_transition_count:
                        spa_transition_count[tuple([list_line[2], previous_tag])]+=1
                    else:
                        spa_transition_count[tuple([list_line[2], previous_tag])]=1
                if list_line[1] in ['eng','eng&spa','UNK']:
                    if list_line[2] in eng_transition_count:
                        eng_transition_count[tuple([list_line[2], previous_tag])]+=1
                    else:
                        eng_transition_count[tuple([list_line[2], previous_tag])]=1
            previous_tag = list_line[2]
            previouse_word_language = list_line[1]
            if tuple([list_line[2], list_line[0]]) in emission_count:  # counting emission probability
                emission_count[tuple([list_line[2], list_line[0]])] += 1
                
            else:
                emission_count[tuple([list_line[2], list_line[0]])] = 1
                
            if list_line[1] in ['spa','eng&spa','UNK'] and tuple([list_line[2], list_line[0]]) in spanish_emission_count:
                    spanish_emission_count[tuple([list_line[2], list_line[0]])] += 1
            elif list_line[1] in ['spa','eng&spa','UNK']:
                    spanish_emission_count[tuple([list_line[2], list_line[0]])] = 1
                    
            if list_line[1] in ['eng','eng&spa','UNK'] and  tuple([list_line[2], list_line[0]]) in english_emission_count:
                    english_emission_count[tuple([list_line[2], list_line[0]])] += 1
            elif list_line[1] in ['eng','eng&spa','UNK']:
                    english_emission_count[tuple([list_line[2], list_line[0]])] = 1


    if tuple(['END', previous_tag]) in transition_count:
        transition_count[tuple(['END', previous_tag])] += 1
    else:
        transition_count[tuple(['END', previous_tag])] = 1
    #COUNtING PROBABILITY FOR UNSEEN WORDS WITH WORDS OCCURED ONLY ONCE
    unseen_dictionary={}
    for word,value in word_counting.items():
        if value == 1:#this word occures only once so we need to add (tag,unseen_word) into dictionary
            for key,v in emission_count.items():
                if word in key:
                    if tuple([key[0],'unseen_word']) in unseen_dictionary:
                        unseen_dictionary[tuple([key[0],'unseen_word'])]+=1
                    else:
                        unseen_dictionary[tuple([key[0],'unseen_word'])]=1
    emission_count.update(unseen_dictionary)
    return tag_count_dictionary,spa_tag_count,eng_tag_count, transition_count, emission_count, \
    sentence_counter,word_counting,spanish_emission_count,english_emission_count,  spa_transition_count,\
    eng_transition_count,eng_word_counting,spa_word_counting


def viterbi(test_data_path, state_graph, tag_count,spa_tag_count,eng_tag_count, transition_count, emission_count,spanish_emission_count,english_emission_count,\
 sentence_counter,word_counting,spa_transition_count, eng_transition_count,eng_word_counting,spa_word_counting):
    # reading data from test file line by line and compute needed viterbi values for each value
    sentences = []  # aka OBSERVATIONS
    language_in_sentence = []  # just easier to print that in the file
    language = []
    sentence = []
    good_turing_N=0
    for key,value in word_counting.items():
        good_turing_N +=value
    for line in open(test_data_path, 'r',encoding='UTF-8'):
        if len(line.split()) == 0:  # end of sentence so we can start our calculation for this sentence
            sentences.append(sentence)
            language_in_sentence.append(language)
            sentence = []
            language = []
        else:
            sentence.append(line.split()[0])
            language.append(line.split()[1])
    # taking care of the last sentence
    if sentence:
        sentences.append(sentence)
        language_in_sentence.append(language)

    orig_stdout = sys.stdout
    fout = open('submission.txt', 'w')
    sys.stdout = fout

    for sentence, lang in zip(sentences, language_in_sentence):
        k = len(sentence)  # number of time steps
        viterbi_matrix = {}
        backpointer = {}
        if lang[0] in ['spa','eng&spa']:
            emission = spanish_emission_count
            tag_variable = tag_count
            transition_variable = transition_count
            word_counting_variable = spa_word_counting
        elif lang[0] in ['eng','eng&spa']:
            emission = english_emission_count
            tag_variable = tag_count
            transition_variable = transition_count
            word_counting_variable = eng_word_counting
        else: 
            emission = emission_count #all spa and eng words together
            tag_variable = tag_count
            transition_variable = transition_count
            word_counting_variable = word_counting
        for state in state_graph:  # initialization step
            state_value = []
            if tuple([state, 'START']) in transition_variable:
                a = transition_variable[tuple([state, 'START'])] / sentence_counter
            else:
                a=0
            if tuple([state, sentence[0]]) in emission:  # this word in dictionary
                c = emission[tuple([state, sentence[0]])]
                b = c / tag_variable[state]
            elif (sentence[0] in word_counting_variable):#unseen words handeled in this part
                b=0
            else:  
                if sentence[0][0].isupper() and state == 'PROPN':# if First letter of the word Upper case more likely it will be PROPN
                    b =0.5
                else:
                    if tuple([state,'unseen_word']) in emission_count:    
                        b = emission_count[tuple([state,'unseen_word'])]/tag_count[state]
                    else:
                        if state in tag_count:
                            b = 1/tag_count[state]
                        else:
                            b=0
            if sentence[0] == 'there' and state == 'PRON': #correction by hand
                playing_with_dev_set = 1
            else:
                playing_with_dev_set = 0
            state_value.append(a * b+playing_with_dev_set)
            viterbi_matrix[state] = state_value  # initializing viterbi matrix
            backpointer[state] = ['START']
        # recursion step
        for t in range(1, k):
            aaa=0
            if lang[t]  in ['spa'] and lang[t-1] == 'spa':
                emission = spanish_emission_count
                tag_variable = tag_count
                transition_variable = transition_count
                word_counting_variable = spa_word_counting
            elif lang[t] in ['eng'] and lang[t-1] == 'eng':
                emission = english_emission_count
                tag_variable = tag_count
                transition_variable = transition_count
                word_counting_variable=eng_word_counting
            else: 
                emission = emission_count #all spa and eng words together
                tag_variable = tag_count
                transition_variable = transition_count
                word_counting_variable = word_counting
            for state in state_graph:
                links_value = []  # from this values we are going to choose maximum value
                backtrack_values = []
                for previouse_step_state in state_graph:  # computing new value based on previouse step and choosing
                    # maximum at the end
                    if tuple([state, previouse_step_state]) in transition_variable:  # not new transition between tags
                        a = transition_variable[tuple([state, previouse_step_state])] / tag_variable[previouse_step_state]
                    else:
                        a=0
                    if tuple([state, sentence[t]]) in emission:  # not new word to this tag
                        c = emission[tuple([state, sentence[t]])]
                        b = c / tag_variable[state]
                    elif (sentence[t] in word_counting_variable): 
                        b=0
                    else:   #unseen words handels here
                        if sentence[t][0].isupper() and state == 'PROPN':# if First letter of the word Upper case more likely it will be PROPN
                            b =0.5
                        else:    
                            if tuple([state,'unseen_word']) in emission_count:    
                                b = emission_count[tuple([state,'unseen_word'])]/tag_count[state]
                            else:
                                if state in tag_count:
                                    b = 1/tag_count[state]
                                else:
                                    b=0
                    playing_with_dev_set = 0

                    if sentence[t] in ["'m","'s","'re","are","is", "am","was","were","have","do","did"] :
                        if len(sentence)-1 > t:
                            if ((sentence[t]=="have" and sentence[t+1]=="to") or (sentence[t] in ["do","did"] and sentence[t+1]=="n't")) and state == 'AUX':
                                playing_with_dev_set = 0.01
                            elif len(sentence[t+1])>3:
                                if sentence[t+1][-3:]=="ing" and state == 'AUX':
                                    playing_with_dev_set = 0.01
                    if (sentence[t] == 'no' and state=='ADV') or \
                            (sentence[t] == 'voy' and state=='AUX') or \
                            (sentence[t] == 'about' and state == 'ADP'):
                           # (sentence[t]=='there' and (previouse_step_state in ['CCONJ','SCONJ']) and state == 'PRON'):
                        playing_with_dev_set = 0.01
                    
                    links_value.append(viterbi_matrix[previouse_step_state][t - 1] * a * b+playing_with_dev_set)
                    backtrack_values.append(viterbi_matrix[previouse_step_state][t - 1] * a)
                viterbi_matrix[state].append(max(links_value))
                backpointer[state].append(state_graph[np.argmax(backtrack_values)])  # puts actual tag into backpointer
        # termination step
        links_value = []
        backtrack_values = []
        for state in state_graph:
            #tag_variable = tag_count
            if tuple(['END', state]) in transition_variable:  # not new transition between tags
                a = transition_variable[tuple(['END', state])] / tag_variable[state]
            else:
                a = 0
            links_value.append(viterbi_matrix[state][len(sentence) - 1] * a)
            backtrack_values.append(viterbi_matrix[state][len(sentence) - 1] * a)
        viterbi_matrix['END'] = []
        viterbi_matrix['END'].append(max(links_value))
        backpointer['END'] = []
        backpointer['END'].append(state_graph[np.argmax(backtrack_values)])
        path = backpointer[backpointer['END'][0]][1:]
        path.append(backpointer['END'][0])
        for word, l, tag in zip(sentence, lang, path):
            print(word + '\t' + l + '\t' + tag)

        print()
        '''for key,value in viterbi_matrix.items():
            output_str =''
            output_str = '  '.join(str(round(i,10)) for i in value)
            output_str = str(key)+'       '+output_str
            print(output_str)'''

# return sentences,language_in_sentance,path

def cmpFiles(experimentResultFile, trueResultFile,word_counting):
    experimentResult = []
    trueResult = []
    result = True
    negative_results = 0
    total_number_results = 0
    for line in open(experimentResultFile, 'r'):
        if line.split():
            line_list = []
            for ids in line.split():
                line_list.append(ids)
            experimentResult.append(line_list)
            total_number_results += 1
    set_of_words =set()
    for line in open(trueResultFile, 'r',encoding ='UTF-8'):
        if line.split():
            line_list = []
            for ids in line.split():
                line_list.append(ids)
            set_of_words.add(line_list[0])
            trueResult.append(line_list)

    line_counter=1
    for word in set_of_words:
        for trueLine, expLine in zip(trueResult, experimentResult):   
            if trueLine != expLine and expLine[0]==word:
                negative_results += 1
                print('True = ',trueLine,'Exper = ',expLine,'Line = ',line_counter)
            line_counter+=1
    return round((total_number_results - negative_results) / total_number_results, 4)
'''
def good_turing_smoothing(emission_count,c):#retrund Nc value, c - word occurencies
    frequency =[]
    count=0
    if c==0:
        for i in range(1,51):
            for key,value in emission_count.items():
                if value == i:
                    count+=1
            frequency.append(count)
            count=0
        return frequency
    else:
        for key,value in emission_count.items():
            if value == c:
                count+=1
        if count ==0:
            count = 1
        return count
'''
tag_count,spa_tag_count,eng_tag_count, transition, emission, sentance_count,word_counting,spanish_emission_count,\
english_emission_count,spa_transition_count, eng_transition_count,eng_word_counting,spa_word_counting = \
counting_probabilities('dataset/train.conll')
state_graph = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
               'SCONJ', 'SYM', 'VERB', 'X','UNK']
'''print(eng_tag_count['ADP'])
print(english_emission_count['ADP','a'])
print(english_emission_count['DET','a'])
print(eng_tag_count['DET'])
'''
#print(tuple(['PART','to']),emission[tuple(['PART','to'])])
#print(emission[tuple(['PART','to'])])
viterbi('dataset/test.conll', state_graph, tag_count,spa_tag_count,eng_tag_count, transition,\
emission,spanish_emission_count,english_emission_count, sentance_count,word_counting,spa_transition_count,eng_transition_count,eng_word_counting,spa_word_counting)
#viterbi('dataset/small_dataset.txt', state_graph, tag_count,spa_tag_count,eng_tag_count, transition, emission,spanish_emission_count,english_emission_count,\
#sentance_count,word_counting,spa_transition_count, eng_transition_count,eng_word_counting,spa_word_counting)
'''
orig_stdout = sys.stdout
fout = open('wrongpredictions.txt', 'w')
sys.stdout = fout
print('Accuracy ',cmpFiles('submission.txt','dataset/dev.conll',state_graph))
'''