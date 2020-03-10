import numpy
import re
import pandas

def count_occurence_vector(text, dim=500):
    arr = numpy.zeros(dim, numpy.int32)
    arr[0] = len(text)
    for c in text:
        arr[ord(c)%dim]+=1
    return arr

def char_ignore_mask(chars):
    occ = count_occurence_vector(chars)
    mask = numpy.zeros(occ.shape, numpy.int32)
    mask[occ==0] = 1
    return mask

def find_all_occurences_in_string(pattern, text, lower=True):
    if lower:
        pattern = pattern.lower()
        text = text.lower()
    occurences = [m.start() for m in re.finditer(pattern, text)]
    return occurences

def get_most_frequent(List): 
    dict = {} 
    count, itm = 0, '' 
    for item in reversed(List): 
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count : 
            count, itm = dict[item], item 
    return(itm)

def flatten_column(df, col):
    flat_col = pandas.DataFrame([[i, x] 
               for i, y in df[col].apply(list).iteritems() 
                   for x in y], columns=list(['I',str(col)+'_flat']))
    flat_col = flat_col.set_index('I')
    return df.merge(flat_col, left_index=True, right_index=True)