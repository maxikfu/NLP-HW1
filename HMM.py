
import sys
import codecs
import numpy as np


resultList=[]


def counting_probabilities(file_path):#retruns unigram and bigram counts for probability matrix
	columnCounter=0
	sentance_counter = 0
	previouse_tag = None
	tag_count_dictionary = {}
	transition_count = {} 
	emission_count = {}
	f = codecs.open(file_path, encoding='utf-8')
	for line in f:
		list_line=[]
		for col in line.split():
			list_line.append(col)
		if len(list_line)==0: #sentance ended computing transition from tag to end state
			if tuple(['END',previouse_tag]) in transition_count:
				transition_count[tuple(['END',previouse_tag])]+=1
			else:
				transition_count[tuple(['END',previouse_tag])]=1
			previouse_tag = None # for computing start to tag
		else:
			if list_line[2] in tag_count_dictionary:#counting tag occurencies
				tag_count_dictionary[list_line[2]]+=1 #counting delimiter for emission 
			else:
				tag_count_dictionary[list_line[2]]=1
			if previouse_tag == None: #computing probability from STRAT to tag
				sentance_counter+=1
				if tuple([list_line[2],'START']) in transition_count: #increasing counter
					transition_count[tuple([list_line[2],'START'])]+=1
				else: #creating new one
					transition_count[tuple([list_line[2],'START'])]=1
			else: #we are in the middle of sentance
				if tuple([list_line[2],previouse_tag]) in transition_count:
					transition_count[tuple([list_line[2],previouse_tag])] +=1
				else:
					transition_count[tuple([list_line[2],previouse_tag])] =1
			previouse_tag = list_line[2]
			if tuple([list_line[2],list_line[0]]) in emission_count: #counting emission probability
				emission_count[tuple([list_line[2],list_line[0]])]+=1
			else:
				emission_count[tuple([list_line[2],list_line[0]])]=1
	if tuple(['END',previouse_tag]) in transition_count:
		transition_count[tuple(['END',previouse_tag])]+=1
	else:
		transition_count[tuple(['END',previouse_tag])]=1
	return tag_count_dictionary,transition_count,emission_count,sentance_counter

def viterbi(test_data_path,state_graph,tag_count,transition_count,emission_count,sentance_counter):
	#reading data from test file line by line and compute needed viterbi values for each value
	sentances = []#aka OBSERVATIONS
	language_in_sentance = [] #just easier to print that in the file
	language = []
	sentance = []
	for line in open(test_data_path,'r'):
		if len(line.split()) == 0:#end of sentance so we can start our calculation for this sentance
			sentances.append(sentance)
			language_in_sentance.append(language) 
			sentance = []
			language = []
		else:
			sentance.append(line.split()[0])				
			language.append(line.split()[1])
	#taking care of the last sentance
	sentances.append(sentance)
	language_in_sentance.append(language)
	orig_stdout = sys.stdout
	fout = open('submission.txt', 'w')
	sys.stdout = fout
	for sentance in sentances:
		k = len(sentance)#number of time steps
		viterbi_matrix = {}
		backpointer = {}
		for state in state_graph:#initialization step
			state_value = []
			if tuple([state,'START']) in transition_count:
				a = transition_count[tuple([state,'START'])]/sentance_counter
			else:
				a=0
			if tuple([state,sentance[0]]) in emission_count:#this word in dictionary
				'''print(state,sentance[0])
				print(emission_count[tuple([state,sentance[0]])],tag_count[state])'''
				b = emission_count[tuple([state,sentance[0]])]/tag_count[state]
			else:
				#print('Not in training ',tuple([state,sentance[0]]))
				b=0
			state_value.append(a*b)
			viterbi_matrix[state] = state_value#initializing viterbi matrix
			backpointer[state]=['START']
		#recursion step
		for t in range(1,k):
			for state in state_graph:
				links_value = []#from this values we are going to choose maximum value
				backtrack_values = []
				for previouse_step_state in state_graph:# computing new value based on previouse step and choosing maximum at the end
					if tuple([state,previouse_step_state]) in transition_count: # not new transition between tags
						a = transition_count[tuple([state,previouse_step_state])]/tag_count[previouse_step_state]
					else:
						a = 0
					if tuple([state,sentance[t]]) in emission_count: #not new word to this tag
						b = emission_count[tuple([state,sentance[t]])]/tag_count[state]
					else:
						b = 0
					links_value.append(viterbi_matrix[previouse_step_state][t-1]*a*b)
					backtrack_values.append(viterbi_matrix[previouse_step_state][t-1]*a)
				viterbi_matrix[state].append(max(links_value))
				backpointer[state].append(state_graph[np.argmax(backtrack_values)])#puts actual tag into backpointer
		#termination step
		links_value = []
		backtrack_values = []
		for state in state_graph:
			if tuple(['END',state]) in transition_count: # not new transition between tags
				a = transition_count[tuple(['END',state])]/tag_count[state]
			else:
				a = 0
			links_value.append(viterbi_matrix[state][t]*a)
			backtrack_values.append(viterbi_matrix[state][t]*a)
		viterbi_matrix['END'] = []
		viterbi_matrix['END'].append(max(links_value))
		backpointer['END']=[]
		backpointer['END'].append(state_graph[np.argmax(backtrack_values)])
		path = backpointer[backpointer['END'][0]][1:]
		path.append(backpointer['END'][0])
		
	return sentances,language_in_sentance,path

tag_count,transition,emission,sentance_count =counting_probabilities('dataset/train.conll')
state_graph = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X']
matrix = viterbi('dataset/small_test.txt',state_graph,tag_count,transition,emission,sentance_count)

orig_stdout = sys.stdout
fout = open('out.txt', 'w')
sys.stdout = fout
print(matrix)
#print(transition)
orig_stdout = sys.stdout
fout.close()

