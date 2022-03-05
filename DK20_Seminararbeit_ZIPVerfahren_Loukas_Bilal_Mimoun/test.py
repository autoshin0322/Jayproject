dict_size = 256
dictionary = {chr(i): i for i in range(dict_size)}
uncompressed = "LZWLZ78LZ77LZCLZMWLZAP"

w = ""
result = []
for c in uncompressed:
    wc = w + c                          
    if wc in dictionary:
        w = wc                          
    else:
        result.append(dictionary[w])    
        # Add wc to the dictionary.     
        dictionary[wc] = dict_size      
        dict_size += 1
        w = c

# Output the code for w.
if w:
    result.append(dictionary[w])
print(result)