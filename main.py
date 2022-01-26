import os
import string
import pickle
import hashlib
from collections import defaultdict
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import threading
import time
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

recipes_dir = './recipes'
saved_dict_location = './index/reversed_index.pkl'

title_w = 100
intro_w = 5
ingrident_w = 20
method_w = 10

# The keys of the rev_index are words. The values are a list of pairs, first one is recipe id second is score for that word 

# Weights: Title: 100, ingridents 20, description 10, introduction 10

# convert to lower case, remove punctuation, split, filter out stop words, stem the list of words
# then remove duplicates
def get_keywords(ws: str) -> [str]:
    ws1 = word_tokenize(ws.lower().translate(str.maketrans('', '', string.punctuation)))
    ws2 = [stemmer.stem(x) for x in ws1 if not x in stop_words]
    return list(set(ws2))

def add_line_to_dict(rid,line,dic,weight):
    for word in get_keywords(line):
        if dic[word] != [] and dic[word][-1][0] == rid:
            dic[word][-1][1] += weight
        else:
            dic[word].append([rid,weight])

def assert_sec_title(line,t,recipe):
    if line != t:
        pass#print("Recipe " + recipe + " is incomplete")

def const_empty_list():
    return []

def build_reverse_index():
    word_index = defaultdict(const_empty_list,[])
    id_to_recipie = {} # Map the id to the title of the recipe
    recipe_id = 0
    for recipe in os.listdir(recipes_dir): 
        with open('./recipes/' + recipe, 'r') as lines:
            # Go through the recipe adding each line to the dictionary with the appropriate weight
            line = lines.readline() # The title
            id_to_recipie[recipe_id] = line # Assosiate the recipie id with its title
            add_line_to_dict(recipe_id,line,word_index,title_w)

            assert_sec_title(lines.readline(),"Introduction:\n",recipe)
            line = lines.readline()
            add_line_to_dict(recipe_id,line,word_index,intro_w)

            assert_sec_title(lines.readline(),"Ingredients:\n",recipe)
            line = lines.readline()
            add_line_to_dict(recipe_id,line,word_index,ingrident_w)

            assert_sec_title(lines.readline(),"Method:\n",recipe)
            line = lines.readline()
            add_line_to_dict(recipe_id,line,word_index,method_w)

        recipe_id += 1
    
    return (word_index,id_to_recipie)

# Algorithm to merge the lists of (recipe_id,score) pairs from the different words of the query
def merge_ordered_lists_of_pairs(xss): # merge the scores
    if xss == []:
        return []
    ys = xss[0] # set first list to result list
    xsi = 1
    while xsi < len(xss): # loop through outer list
        yi = 0
        xi = 0
        while xi < len(xss[xsi]) and yi < len(ys): # loop through ys and next xs simutaniously
            if xss[xsi][xi][0] == ys[yi][0]: # if recipe ids match, add scores together
                ys[yi][1] += xss[xsi][xi][1]
                xi += 1
            elif ys[yi][0] > xss[xsi][xi][0]: # if we havent got recipe in result list, add it
                ys.insert(yi,xss[xsi][xi])
                xi += 1
            yi += 1
        ys += xss[xsi][xi:] # add the remainder of xs to ys
        xsi += 1
    return ys

def number_items(xs):
    return list(map(lambda x: str(x[0]) + ". " + x[1], zip(range(1,len(xs)+1),xs)))

def perform_search(query: str, rev_index = None, ids=None) -> [str]:
    if rev_index == None:
        with open(saved_dict_location, 'rb') as f:
            (rev_index,ids) = pickle.load(f)
    pure_query = get_keywords(query)
    recipes_and_scores = list(map(lambda x:rev_index[x], pure_query))
    merged_r_and_s = merge_ordered_lists_of_pairs(recipes_and_scores)
    top_recipe_ids = sorted(merged_r_and_s,key=lambda x:x[1],reverse=True)[:10]
    top_recipes = number_items(list(map(lambda x: ids[x[0]] + "  Score: " + str(x[1]), top_recipe_ids)))
    return top_recipes

def search_loop(reload_index,check_index,index_lock):
    reload_index.wait() # wait for inital reload
    while True:
        if reload_index.is_set(): # check if we need to reload
            index_lock.acquire()
            with open(saved_dict_location, 'rb') as f:
                (rev_index,ids,_) = pickle.load(f)
            index_lock.release()
            reload_index.clear() # set reload index to clear as we have reloaded
            print("New index loaded!")
        print("Please enter search query: ")
        query = input()
        tic = time.perf_counter()
        top_recipes = perform_search(query,rev_index,ids)
        toc = time.perf_counter()
        print("The results are in!:\n")
        print("\n".join(top_recipes))
        print(f"\nsearched in {toc - tic:0.4f} seconds\n\n")
        check_index.set() # tell the other thread to check the index, just in case its changed recently

def hash_files(file_list):
    return hashlib.sha1((" ".join(file_list)).encode()).hexdigest()

def index_loop(reload_index,check_index,index_lock):
    index_lock.acquire()
    recipes_hash = 0
    if os.path.isfile(saved_dict_location):
        with open(saved_dict_location, 'rb') as f:
            (_,_,recipes_hash) = pickle.load(f) # get the inital recipe hash
        if hash_files(os.listdir(recipes_dir)) == recipes_hash:
            reload_index.set()
    index_lock.release()
    

    while True:
        check_index.wait() # only check after a search so as not to waste looping here
        current_recipes_hash = hash_files(os.listdir(recipes_dir))

        if current_recipes_hash != recipes_hash:
            index_lock.acquire()
            
            #print("\nRefreshing index\n")
            (word_index,id_to_recipie) = build_reverse_index() # build new index
            #print("\nIndex refrehsed!1!!\n")
            with open(saved_dict_location, 'wb') as f:
                pickle.dump((word_index,id_to_recipie,current_recipes_hash), f) # save index to file
            recipes_hash = current_recipes_hash

            index_lock.release()
            reload_index.set() # tell search loop that it needs to reload
        check_index.clear() # reset the check index since we have just checked
    
    

def main():
    reload_index = threading.Event()
    check_index = threading.Event()
    index_lock = threading.Lock()

    reload_index.clear()
    check_index.set()

    search = threading.Thread(target=search_loop,args=(reload_index,check_index,index_lock),name='search')
    index = threading.Thread(target=index_loop,args=(reload_index,check_index,index_lock),name='index')
    index.start()
    search.start()



main()