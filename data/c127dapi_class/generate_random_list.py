import sys
import numpy as np
import os
def generate_random_list(number_to_include, directory, out_filename):


	
	store_lines = []
	for file in os.listdir(directory+"/Annotations"):
	    if file.endswith(".xml"):
	        #print(file[:-4])
	        store_lines.append(str(file[:-4]))
	
	
	
	#with open(out_filename) as f:
#		for line in f:
			
	#print(store_lines.__len__())

	#number_to_include = np.ceil(float(fraction)*float(store_lines.__len__()))

	indices_to_use = np.sort(np.random.choice(np.arange(0,store_lines.__len__()), size=int(number_to_include), replace=False))

	outF = open(out_filename+"_n"+str(int(number_to_include))+".txt", "w")
	store_lines = np.array(store_lines)
	textList = list(map(lambda x: x, store_lines[indices_to_use]))
	outF.writelines("%s\n" % l for l in textList)
	outF.close()





if __name__ == "__main__":
    generate_random_list(*sys.argv[1:])