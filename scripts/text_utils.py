import os

bad_books = ["a_room_with_a_view", "ethan_frome", 
            "madame_de_treymes", "persuasion", 
            "sense_and_sensibility", "silas_marner", 
            "summer", "the_emerald_city_of_oz", 
            "the_secret_garden", "daisy_miller",
            "treasure_island","through_the_looking_glass",
            "the_gift_of_the_magi", "the_patchwork_girl_of_oz"]

def read_lines( filename ):
    f = open(filename)
    lines = [ i.strip() for i in f.read().split("\n") if i.strip()!="" ]
    f.close()
    return lines

def write_lines( filename, lines ):
    f = open(filename, "w")
    f.write( "\n".join(lines) )
    f.close()

def transform_cath_filenames( outdir, filename ):
    return os.path.join( outdir, "_".join(filename.split("/")[-2:])+".npy" )

def find_in_list( txt, list_of_txts ):
    for i in list_of_txts:
        if txt.find(i)!=-1:
            return True
    return False