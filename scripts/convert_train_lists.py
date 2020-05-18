import os
import sys

input_filename = sys.argv[1]
wav_location = sys.argv[2]
output_filename = sys.argv[3]

def replace_character( inp, index, final_char ):
	b = list(inp)
	b[index] = final_char
	return "".join(b)

def find_last_occurence( inp, to_find ):
	p = len(inp)-1-inp[::-1].find(to_find)
	return p

lines = open(input_filename).read().split("\n")
lines = [line.split("|") for line in lines]

outlines = []
for line in lines:
	wav_filename = os.path.basename(line[0])
	final_filename = os.path.join( wav_location, replace_character( wav_filename, find_last_occurence(wav_filename, "_"), "/" ).replace(".npy","") )
	outlines.append( "|".join( [ final_filename, line[1], line[2] ] ) )


f = open(output_filename, "w")
f.write("\n".join(outlines))
f.close()