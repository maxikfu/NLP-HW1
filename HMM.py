import sys
import codecs
import numpy as np

resultList = []


def counting_probabilities(file_path):  # returns unigram and bigram counts for probability matrix
    sentence_counter = 0
    previous_tag = None
    tag_count_dictionary = {}
    spa_tag_count={}
    eng_tag_count={}
    transition_count = {}
    emission_count = {}
    spanish_emission_count={}
    english_emission_count={}
    word_counting = {}
    for line in open(file_path,'r',encoding='UTF-8'):
        list_line = []
        for col in line.split():
            list_line.append(col)
        if len(list_line) == 0:  # sentence ended computing transition from tag to end state
            if tuple(['END', previous_tag]) in transition_count:
                transition_count[tuple(['END', previous_tag])] += 1
            else:
                transition_count[tuple(['END', previous_tag])] = 1
            previous_tag = None  # for computing start to tag
        else:
            #for good turing smoothing
            if list_line[0] in word_counting:
                word_counting[list_line[0]]+=1
            else:
                word_counting[list_line[0]]=1


            if list_line[2] in tag_count_dictionary:  # counting tag occurrences
                tag_count_dictionary[list_line[2]] += 1  # counting delimiter for emission
            else:
                tag_count_dictionary[list_line[2]] = 1
            if list_line[1]=='spa':
                if list_line[2] in spa_tag_count:
                    spa_tag_count[list_line[2]]+=1
                else:
                    spa_tag_count[list_line[2]]=1
            elif list_line[1]=='eng':
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
            else:  # we are in the middle of sentence
                if tuple([list_line[2], previous_tag]) in transition_count:
                    transition_count[tuple([list_line[2], previous_tag])] += 1
                else:
                    transition_count[tuple([list_line[2], previous_tag])] = 1
            previous_tag = list_line[2]
            if tuple([list_line[2], list_line[0]]) in emission_count:  # counting emission probability
                emission_count[tuple([list_line[2], list_line[0]])] += 1
                
            else:
                emission_count[tuple([list_line[2], list_line[0]])] = 1
                
            if list_line[1] == 'spa' and tuple([list_line[2], list_line[0]]) in spanish_emission_count:
                    spanish_emission_count[tuple([list_line[2], list_line[0]])] += 1
            elif list_line[1] == 'spa':
                    spanish_emission_count[tuple([list_line[2], list_line[0]])] = 1
                    
            if list_line[1] == 'eng' and  tuple([list_line[2], list_line[0]]) in english_emission_count:
                    english_emission_count[tuple([list_line[2], list_line[0]])] += 1
            elif list_line[1] == 'eng':
                    english_emission_count[tuple([list_line[2], list_line[0]])] = 1


    if tuple(['END', previous_tag]) in transition_count:
        transition_count[tuple(['END', previous_tag])] += 1
    else:
        transition_count[tuple(['END', previous_tag])] = 1
    return tag_count_dictionary,spa_tag_count,eng_tag_count, transition_count, emission_count, \
    sentence_counter,word_counting,spanish_emission_count,english_emission_count


def viterbi(test_data_path, state_graph, tag_count,spa_tag_count,eng_tag_count, transition_count, emission_count,spanish_emission_count,english_emission_count,\
 sentence_counter,word_counting):
    # reading data from test file line by line and compute needed viterbi values for each value
    sentences = []  # aka OBSERVATIONS
    language_in_sentence = []  # just easier to print that in the file
    language = []
    sentence = []
    good_turing_N=0
    for key,value in word_counting.items():
        good_turing_N +=value
    good_turing_N_1 = good_turing_smoothing(word_counting,1)
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
        for state in state_graph:  # initialization step
            state_value = []
            if tuple([state, 'START']) in transition_count:
                a = transition_count[tuple([state, 'START'])] / sentence_counter
            else:
                a=0
            if lang[0] == 'spa':
                emission = spanish_emission_count
                tag_variable = spa_tag_count
            elif lang[0]=='eng':
                emission = english_emission_count
                tag_variable = eng_tag_count
            else: 
                emission = emission_count #all spa and eng words together
                tag_variable = tag_count
            if tuple([state, sentence[0]]) in emission:  # this word in dictionary
                c = emission[tuple([state, sentence[0]])]
                #b_new = ((c+1)*good_turing_smoothing(word_counting,c+1)/good_turing_smoothing(word_counting,c)) / tag_count[state]
                b = c / tag_variable[state]
            else:#unseen words handeled in this part  
                if sentence[0][0].isupper() and state == 'PROPN':# if First letter of the word Upper case more likely it will be PROPN
                    b =good_turing_N_1/good_turing_N
                else:    
                    b =good_turing_N_1/good_turing_N*0.00005
            if sentence[0] == 'there' and state == 'PRON': #correction by hand
                playing_with_dev_set = 1
            else:
                playing_with_dev_set = 0
            state_value.append(a * b+playing_with_dev_set)
            viterbi_matrix[state] = state_value  # initializing viterbi matrix
            backpointer[state] = ['START']
        # recursion step
        for t in range(1, k):
            for state in state_graph:
                links_value = []  # from this values we are going to choose maximum value
                backtrack_values = []
                for previouse_step_state in state_graph:  # computing new value based on previouse step and choosing
                    # maximum at the end
                    if tuple([state, previouse_step_state]) in transition_count:  # not new transition between tags
                        a = transition_count[tuple([state, previouse_step_state])] / tag_count[previouse_step_state]
                    else:
                        a=0
                    if lang[t] == 'spa':
                        emission = spanish_emission_count
                        tag_variable = spa_tag_count
                    elif lang[t]=='eng':
                        emission = english_emission_count
                        tag_variable = eng_tag_count
                    else: 
                        emission = emission_count #all spa and eng words together
                        tag_variable = tag_count
                    if tuple([state, sentence[t]]) in emission:  # not new word to this tag
                        c = emission[tuple([state, sentence[t]])]
                        #b_new = ((c+1)*good_turing_smoothing(word_counting,c+1)/good_turing_smoothing(word_counting,c)) / tag_count[state]
                        b = c / tag_variable[state]
                    else: #unseen words handels here
                        if sentence[t][0].isupper() and state == 'PROPN':# if First letter of the word Upper case more likely it will be PROPN
                            b =good_turing_N_1/good_turing_N
                        else:    
                            b =good_turing_N_1/good_turing_N*0.00005
                    playing_with_dev_set = 0
                    if sentence[t] in ["'m","'s","'re","are","is", "am","was","were","have","do"] :
                        if len(sentence)-1 > t:
                            if ((sentence[t]=="have" and sentence[t+1]=="to") or (sentence[t]=="do" and sentence[t+1]=="n't")) and state == 'AUX':
                                playing_with_dev_set = 0.005
                            elif len(sentence[t+1])>3:
                                if sentence[t+1][-3:]=="ing" and state == 'AUX':
                                    playing_with_dev_set = 0.005
                    if (sentence[t] == 'no' and state=='ADV') or \
                            (sentence[t] == 'voy' and state=='AUX') or \
                            (sentence[t] == 'about' and state == 'ADP'):
                           # (sentence[t]=='there' and (previouse_step_state in ['CCONJ','SCONJ']) and state == 'PRON'):
                        playing_with_dev_set = 0.005
                    
                    links_value.append(viterbi_matrix[previouse_step_state][t - 1] * a * b+playing_with_dev_set)
                    backtrack_values.append(viterbi_matrix[previouse_step_state][t - 1] * a)
                viterbi_matrix[state].append(max(links_value))
                backpointer[state].append(state_graph[np.argmax(backtrack_values)])  # puts actual tag into backpointer
        # termination step
        links_value = []
        backtrack_values = []
        for state in state_graph:
            if tuple(['END', state]) in transition_count:  # not new transition between tags
                a = transition_count[tuple(['END', state])] / tag_count[state]
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

tag_count,spa_tag_count,eng_tag_count, transition, emission, sentance_count,word_counting,spanish_emission_count,english_emission_count = \
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
emission,spanish_emission_count,english_emission_count, sentance_count,word_counting)
#viterbi('dataset/small_dataset.txt', state_graph, tag_count,spa_tag_count,eng_tag_count, transition, emission,spanish_emission_count,english_emission_count,\
 #sentance_count,word_counting)
'''
orig_stdout = sys.stdout
fout = open('wrongpredictions.txt', 'w')
sys.stdout = fout
print('Accuracy ',cmpFiles('submission.txt','dataset/dev.conll',state_graph))
'''