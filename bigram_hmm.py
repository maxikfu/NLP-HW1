import sys
import numpy as np

def counting_tri_bi_uni_grams(training_file_path):
    sentence_counter = 0
    previous_tag = None
    tag_count_dictionary = {}
    tag_count_dictionary['START'] = 0
    tag_count_dictionary['END'] = 0
    spa_tag_count={}
    eng_tag_count= {}
    transition_count = {}
    emission_count = {}
    word_counting = {}
    token_language_count = {}
    spa_emission_count = {}
    eng_emission_count = {}
    tag1 = None
    tag2 = None
    tag3 = None
    for line in open(training_file_path, 'r', encoding='UTF-8'):
        list_line = []
        for col in line.split():
            list_line.append(col)
        ############################computing bigrams and unigrams for tags#######################################
        if len(list_line) == 0:  # sentence ended computing transition from tag to end state
            if tuple([previous_tag,'END']) in transition_count:
                transition_count[tuple([previous_tag,'END'])] += 1
            else:
                transition_count[tuple([previous_tag,'END'])] = 1
            tag3 = 'END'
            if tag1 is not None and tag2 is not None and tag3 is not None:
                if tuple([tag1, tag2, tag3]) in transition_count:
                    transition_count[tuple([tag1, tag2, tag3])] += 1
                else:
                    transition_count[tuple([tag1, tag2, tag3])] = 1
            tag1 = None
            tag2 = None
            tag3 = None
            previous_tag = None  # for computing start to tag
        else:
            # for smoothing

            if list_line[0] in word_counting:
                word_counting[list_line[0]] += 1
            else:
                word_counting[list_line[0]] = 1
            tu = tuple([list_line[0], list_line[1]])
            if tu in token_language_count:
                token_language_count[tu] += 1
            else:
                token_language_count[tu] = 1
            if list_line[2] in tag_count_dictionary:  # counting tag occurrences
                tag_count_dictionary[list_line[2]] += 1  # counting delimiter for emission
            else:
                tag_count_dictionary[list_line[2]] = 1
#spa and eng tag count separetly
            if list_line[1] in ['spa']:
                if list_line[2] in spa_tag_count:
                    spa_tag_count[list_line[2]] += 1
                else:
                    spa_tag_count[list_line[2]] = 1
            if list_line[1] in ['eng']:
                if list_line[2] in eng_tag_count:
                    eng_tag_count[list_line[2]] += 1
                else:
                    eng_tag_count[list_line[2]] = 1

            if previous_tag is None:  # computing probability from START to tag
                tag1 = 'START'
                tag2 = list_line[2]
                sentence_counter += 1
                tag_count_dictionary['START'] += 1
                tag_count_dictionary['END'] += 1
                if tuple(['START', list_line[2]]) in transition_count:  # increasing counter
                    transition_count[tuple(['START', list_line[2]])] += 1
                else:  # creating new one
                    transition_count[tuple(['START', list_line[2]])] = 1
            else:  # we are in the middle of sentence
                tag3 = list_line[2]
                if tuple([previous_tag,list_line[2]]) in transition_count:
                    transition_count[tuple([previous_tag,list_line[2]])] += 1
                else:
                    transition_count[tuple([previous_tag,list_line[2]])] = 1
                if tag1 is not None and tag2 is not None and tag3 is not None:
                    if tuple([tag1, tag2, tag3]) in transition_count:
                        transition_count[tuple([tag1, tag2, tag3])] += 1
                    else:
                        transition_count[tuple([tag1, tag2, tag3])] = 1
                    tag1 = tag2
                    tag2 = tag3
                    tag3 = None

            previous_tag = list_line[2]
            if tuple([list_line[2], list_line[0]]) in emission_count:  # counting emission probability
                emission_count[tuple([list_line[2], list_line[0]])] += 1
            else:
                emission_count[tuple([list_line[2], list_line[0]])] = 1
#if we meet word what starts with upper case it amy be a name, then probability its PROPN bigger
            if list_line[0][0].isupper() and len(list_line[0]) > 1:
                if tuple([list_line[2], 'name']) in emission_count:  # counting emission probability
                    emission_count[tuple([list_line[2], 'name'])] += 1
                else:
                    emission_count[tuple([list_line[2], 'name'])] = 1
#counting for spa and eng words separate
            if list_line[1] in ['spa'] and tuple([list_line[2], list_line[0]]) in spa_emission_count:
                spa_emission_count[tuple([list_line[2], list_line[0]])] += 1
            elif list_line[1] in ['spa']:
                spa_emission_count[tuple([list_line[2], list_line[0]])] = 1

            if list_line[1] in ['eng'] and tuple([list_line[2], list_line[0]]) in eng_emission_count:
                eng_emission_count[tuple([list_line[2], list_line[0]])] += 1
            elif list_line[1] in ['eng']:
                eng_emission_count[tuple([list_line[2], list_line[0]])] = 1
    if tuple(['END', previous_tag]) in transition_count:
        transition_count[tuple([previous_tag, 'END'])] += 1
    else:
        transition_count[tuple([previous_tag,'END'])] = 1
    tag3 = 'END'
    if tag1 is not None and tag2 is not None and tag3 is not None:
        if tuple([tag1, tag2, tag3]) in transition_count:
            transition_count[tuple([tag1, tag2, tag3])] += 1
        else:
            transition_count[tuple([tag1, tag2, tag3])] = 1
    # COUNtING PROBABILITY FOR UNSEEN WORDS WITH WORDS OCCURRED ONLY ONCE
    unseen_dictionary = {}
    for word, value in word_counting.items():
        if value == 1:  # this word occurs only once so we need to add (tag,unseen_word) into dictionary
            for key, v in emission_count.items():
                if word in key:
                    if tuple([key[0], 'unseen_word']) in unseen_dictionary:
                        unseen_dictionary[tuple([key[0], 'unseen_word'])] += 1
                    else:
                        unseen_dictionary[tuple([key[0], 'unseen_word'])] = 1
    emission_count.update(unseen_dictionary)
    return emission_count,spa_emission_count,eng_emission_count,tag_count_dictionary,spa_tag_count,\
           eng_tag_count,transition_count, word_counting, sentence_counter,token_language_count


def deleted_interpolation(transition_matrix,tag_count ,token_dictionary):
    lambda_vector = [0, 0, 0]
    list_variable =[]
    n = 0
    for k, v in token_dictionary.items():
        n += v
    for key,value in transition_matrix.items():
        if len(key) == 3 and value > 0:
            if transition_matrix[tuple([key[0],key[1]])]>1:
                list_variable.append((transition_matrix[key]-1)/(transition_matrix[tuple([key[0],key[1]])]-1))
            else:
                list_variable.append(0)
            if tag_count[key[0]] > 1:
                list_variable.append((transition_matrix[tuple([key[1],key[2]])]-1)/(tag_count[key[1]]-1))
            else:
                list_variable.append(0)

            list_variable.append((tag_count[key[2]]-1)/(n-1))
            max_index = np.argmax(list_variable)
            list_variable = []
            if max_index == 0:
                lambda_vector[0] += value
            elif max_index == 1:
                lambda_vector[1] += value
            else:
                lambda_vector[2] += value
    #normalization step
    norm_lambda = [float(i) / sum(lambda_vector) for i in lambda_vector]
    return norm_lambda


def choosing_language(lang,emis,spa_emis,eng_emis,tag,spa_tag,eng_tag):
    lan_emission = None
    lan_tag = None
    if lang == 'eng':
        lan_emission = eng_emis
        lan_tag = eng_tag
    elif lang == 'spa':
        lan_emission = spa_emis
        lan_tag = spa_tag
    else:
        lan_emission = emis
        lan_tag = tag
    return lan_emission


def viterbi(test_data_path, state_graph, tag_count, transition_count, emission_count, word_counting,norm_lambda,\
            spa_emiss,eng_emiss,spa_tag,eng_tag,token_language):
    sentence_counter = tag_count['START']
    sentences = []  # aka OBSERVATIONS
    language_in_sentence = []  # just easier to print that in the file
    language = []
    sentence = []
    n = 0
    for k, v in word_counting.items():
        n += v
    for line in open(test_data_path, 'r', encoding='UTF-8'):
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
#choosing dictionary based on language
        emission = choosing_language(lang[0],emission_count,spa_emiss,eng_emiss,tag_count,spa_tag,eng_tag)
        #emission = emission_count  # all spa and eng words together
        tag_variable = tag_count
        transition_variable = transition_count
        word_counting_variable = word_counting
        for state in state_graph:  # initialization step
            state_value = []
            if tuple(['START',state]) in transition_variable:
                a = transition_variable[tuple(['START',state])] / sentence_counter
            else:
                a = 0
            if tuple([state, sentence[0]]) in emission:  # this word in dictionary
                c = emission[tuple([state, sentence[0]])]
                b = c / tag_variable[state]
            elif tuple([sentence[0],lang[0]]) in token_language:
#toa avoid 0 when word in training set marked as different language but exists in general
                b = 0
            else:#unseen words
                if sentence[0][0].isupper():  # if First letter of the word Upper case more likely it will be PROPN
                    if tuple([state,'name']) in emission_count:
                        b = emission_count[tuple([state, 'name'])] / tag_variable[state]
                    else:
                        b = 0
                else:
                    if tuple([state, 'unseen_word']) in emission_count:
                        b = emission_count[tuple([state, 'unseen_word'])] / tag_variable[state]
                    else:
                        if state in tag_variable:
                            b = 1 / tag_variable[state]
                        else:
                            b = 0
            state_value.append(a * b)
            viterbi_matrix[state] = state_value  # initializing viterbi matrix
            backpointer[state] = ['START']
        # recursion step
        state_graph_extended = ['START'] + state_graph
        for t in range(1, k):
            aaa = 0
#choosing dictionary based on language
            emission = choosing_language(lang[t], emission_count, spa_emiss, eng_emiss, tag_count,\
                                                       spa_tag, eng_tag)
            for state in state_graph:
                links_value = []  # from this values we are going to choose maximum value
                backtrack_values = []
                for previous_step_state in state_graph:  # computing new value based on previouse step and choosing
                    # maximum at the end
                    if tuple([previous_step_state,state]) in transition_variable:  # not new transition between tags
                        a = transition_variable[tuple([previous_step_state,state])] / tag_variable[
                            previous_step_state]
                    else:
                        a = 0
                    if tuple([state, sentence[t]]) in emission:  # not new word to this tag
                        c = emission[tuple([state, sentence[t]])]
                        b = c / tag_variable[state]
                    elif tuple([sentence[t], lang[t]]) in token_language:
                        # avoiding if languages got messed up
                        b = 0
                    else:  # unseen words handels here
                        if sentence[t][0].isupper():
                            if tuple([state,'name']) in emission_count:
                                b = emission_count[tuple([state,'name'])] / tag_variable[state]
                            else:
                                b=0
                        else:
                            if tuple([state, 'unseen_word']) in emission_count:
                                b = emission_count[tuple([state, 'unseen_word'])] / tag_variable[state]
                            else:
                                if state in tag_variable:
                                    b = 1 / tag_variable[state]
                                else:
                                    b = 0
                    playing_with_dev_set = 0

                    links_value.append(viterbi_matrix[previous_step_state][t - 1] * a * b + playing_with_dev_set)
                    backtrack_values.append(viterbi_matrix[previous_step_state][t - 1] * a)
                viterbi_matrix[state].append(max(links_value))
                backpointer[state].append(state_graph[np.argmax(backtrack_values)])
        # termination step
        links_value = []
        backtrack_values = []

        for state in state_graph:
            # tag_variable = tag_count
            if tuple([state,'END']) in transition_variable:  # not new transition between tags
                a = transition_variable[tuple([state,'END'])] / tag_variable[state]
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


def cmpFiles(experiment_result_file, true_result_file):
    experiment_result = []
    true_result = []
    result = True
    negative_results = 0
    total_number_results = 0
    for line in open(experiment_result_file, 'r'):
        if line.split():
            line_list = []
            for ids in line.split():
                line_list.append(ids)
            experiment_result.append(line_list)
            total_number_results += 1
    set_of_words =set()
    for line in open(true_result_file, 'r',encoding ='UTF-8'):
        if line.split():
            line_list = []
            for ids in line.split():
                line_list.append(ids)
            set_of_words.add(line_list[0])
            true_result.append(line_list)
    line_counter=1
    for word in set_of_words:
        for trueLine, expLine in zip(true_result, experiment_result):
            if trueLine != expLine and expLine[0] == word:
                negative_results += 1
                print('True = ',trueLine,'Exper = ',expLine, 'Line = ',line_counter)
            line_counter += 1
    return round((total_number_results - negative_results) / total_number_results, 4)



emission_count,spa_emiss,eng_emiss,tag_count_dictionary,spa_tag,eng_tag, transition_count, word_counting, sentence_counter,tok_len = counting_tri_bi_uni_grams(
    'dataset/train.conll')
state_graph = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
               'SCONJ', 'SYM', 'VERB', 'X', 'UNK']
lambda_param = deleted_interpolation(transition_count,tag_count_dictionary, word_counting)
viterbi('dataset/dev.conll', state_graph, tag_count_dictionary, transition_count,emission_count, word_counting, \
        lambda_param, spa_emiss,eng_emiss,spa_tag,eng_tag,tok_len)

orig_stdout = sys.stdout
fout = open('wrongpredictions.txt', 'w')
sys.stdout = fout
print('Accuracy ',cmpFiles('submission.txt','dataset/dev.conll'))
