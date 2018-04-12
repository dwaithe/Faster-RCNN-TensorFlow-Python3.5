import sys
import numpy as np
import os
from numpy.random import RandomState

def generate_random_list(number_to_include, directory):


	
	store_lines = []
	for file in os.listdir(directory+"/Annotations"):
	    if file.endswith(".xml"):
	        #print(file[:-4])
	        store_lines.append(str(file[:-4]))
	
	
	
	#with open(out_filename) as f:
#		for line in f:
			
	#print(store_lines.__len__())
	np.random.seed(seed=500)
	#number_to_include = np.ceil(float(fraction)*float(store_lines.__len__()))

	indices_to_use = np.random.choice(np.arange(0,store_lines.__len__()), size=store_lines.__len__(), replace=False)
	split  = store_lines.__len__()//2
	training_list = np.sort(indices_to_use[:split])
	test_list = np.sort(indices_to_use[split:])

	outF = open("train_n"+str(int(number_to_include))+".txt", "w")
	store_lines = np.array(store_lines)
	textList = list(map(lambda x: x, store_lines[training_list[:int(number_to_include)]]))
	outF.writelines("%s\n" % l for l in textList)
	outF.close()

	outF = open("test_n"+str(int(number_to_include))+".txt", "w")
	textList = list(map(lambda x: x, store_lines[test_list[:int(number_to_include)]]))
	outF.writelines("%s\n" % l for l in textList)
	outF.close()





if __name__ == "__main__":
    generate_random_list(*sys.argv[1:])