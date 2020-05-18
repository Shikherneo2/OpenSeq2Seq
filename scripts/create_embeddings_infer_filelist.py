import os
import sys

tacotron_filelist_location = sys.argv[1]
output_file = sys.argv[2]

raw_lines = open(tacotron_filelist_location).read().split("\n")
lines = [ i.strip().split("|") for i in raw_lines if i.strip()!="" ]

out_lines = []
for index,line in enumerate(lines):
	line[-1] = str(index+1)
	out_lines.append( "|".join( line ) )

fout = open(output_file, "w")
fout.write( "\n".join(out_lines) )
fout.close()