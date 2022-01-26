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

# Solution:
# We have two threads running concurrently. The searcher and the index builder.

# The index builder looks for an exsiting index in saved_dict_location, and if it doesnt find one
# it builds a new reversed index, i.e. for every word we construct a list of the recipes that it 
# appears in along with a score to indicate how relevent it is the that recipe. The index builder
# runs in the background and writes its new index to the file when its done

# The search thread starts by waiting for the index builder to tell it there is a current index
# ready for it to read (this may be immidate is nothing has changed since last time this program
# has run). It uses the index to find relevent recipes and their scores for each word in a query.
# It merges these lists of recipes and scores with my merge_ordered_lists_of_pairs algorithm. The
# top ten recipes are then displayed to the user and an event is triggered to tell the index builder
# to check if the files have changed, just in case they have since the last search.   

# This model could easily be scaled to as many search threads as are needed with bearly any changes

# Assumptions I am making:
# - The time to build the index doesnt matter too much, since it can be done in the background (we use
#   a lot of pre processing on each line in the recipe, e.g. the nltk stemmer)
# - If the recipes have changed their file names have changed too (I used a hash of the file names
#   in the recipe directory to check if the index needs updating)
#   




# Importance of where a word appears, if in title it has weight 100
# Higher scores are given more prominance
title_w = 100
intro_w = 5
ingrident_w = 20
method_w = 10


# convert to lower case, remove punctuation, split, filter out stop words, stem the list of words
# then remove duplicates
def get_keywords(ws: str) -> [str]:
    ws1 = word_tokenize(ws.lower().translate(str.maketrans('', '', string.punctuation)))
    ws2 = [stemmer.stem(x) for x in ws1 if not x in stop_words]
    return list(set(ws2))

# add all the words in a line to a dict
def add_line_to_dict(rid,line,dic,weight):
    for word in get_keywords(line):
        # check if last recipe added to the words index was this one
        # if so, add the weight on to its score
        if dic[word] != [] and dic[word][-1][0] == rid:
            dic[word][-1][1] += weight
        # If the recipe doesnt already appear under the word, add it
        else:
            dic[word].append([rid,weight])

# Theres only one incorrecly formatted recipe (blueberry-friands.txt) which triggers this print
# so for now I have disabled it
def assert_sec_title(line,t,recipe):
    if line != t:
        # print("Recipe " + recipe + " is incomplete")
        pass

# I can't use a lambda for this becuase lambdas cant be pickled
def const_empty_list():
    return []

def build_reverse_index():
    word_index = defaultdict(const_empty_list,[]) # A dictionary with deafult value []
    id_to_recipie = {} # Map the id to the title of the recipe
    recipe_id = 0
    for recipe in os.listdir(recipes_dir): 
        with open('./recipes/' + recipe, 'r') as lines:
            # Go through the recipe adding each line to the dictionary with the appropriate weight
            line = lines.readline() # The title
            id_to_recipie[recipe_id] = line # Assosiate the recipie id with its title
            add_line_to_dict(recipe_id,line,word_index,title_w)

            assert_sec_title(lines.readline(),"Introduction:\n",recipe)
            add_line_to_dict(recipe_id,lines.readline(),word_index,intro_w)

            assert_sec_title(lines.readline(),"Ingredients:\n",recipe)
            add_line_to_dict(recipe_id,lines.readline(),word_index,ingrident_w)

            assert_sec_title(lines.readline(),"Method:\n",recipe)
            add_line_to_dict(recipe_id,lines.readline(),word_index,method_w)

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

# Number the results (e.g. 1. Spam Pudding)
def number_items(xs):
    return list(map(lambda x: str(x[0]) + ". " + x[1], zip(range(1,len(xs)+1),xs)))

def perform_search(query: str) -> [str]:
    pure_query = get_keywords(query) # clean query
    recipes_and_scores = list(map(lambda x:rev_index[x], pure_query)) # use the rev index on the query
    merged_r_and_s = merge_ordered_lists_of_pairs(recipes_and_scores) # merge the results for each seperate word
    top_recipe_ids = sorted(merged_r_and_s,key=lambda x:x[1],reverse=True)[:10] # sort the results and take top 10
    top_recipes = number_items(list(map(lambda x: ids[x[0]] + "  Score: " + str(x[1]), top_recipe_ids))) # format the output
    return top_recipes

# Search thread
def search_loop(reload_index,check_index,index_lock):
    reload_index.wait() # wait for inital all clear from index_loop
    while True:
        if reload_index.is_set(): # check if we need to reload
            index_lock.acquire()
            with open(saved_dict_location, 'rb') as f:
                (rev_index,ids,_) = pickle.load(f)
            index_lock.release()
            reload_index.clear() # set reload index to clear as we have reloaded
            print("New index loaded!!!!!")
        print("Please enter search query: ")
        query = input()
        tic = time.perf_counter()
        top_recipes = perform_search(query,rev_index,ids)
        toc = time.perf_counter()
        print("The results are in!:\n")
        print("\n".join(top_recipes))
        print(f"\nsearched in {toc - tic:0.4f} seconds\n\n")
        check_index.set() # tell the index_loop to check the index, just in case its changed recently

# I cant use the default python hash function as it uses a random seed for each new run
def hash_files(file_list):
    return hashlib.sha1((" ".join(file_list)).encode()).hexdigest()

# Index builder thread
def index_loop(reload_index,check_index,index_lock):
    index_lock.acquire()
    recipes_hash = 0
    if os.path.isfile(saved_dict_location): # check if theres already an index stored
        with open(saved_dict_location, 'rb') as f:
            (_,_,recipes_hash) = pickle.load(f) # get the inital recipe hash
        if hash_files(os.listdir(recipes_dir)) == recipes_hash: # check the stored index is current
            reload_index.set() # if it is current, give all clear to search_loop
    index_lock.release()
    
    while True:
        check_index.wait() # only check after a search so as not to waste looping here
        current_recipes_hash = hash_files(os.listdir(recipes_dir))

        if current_recipes_hash != recipes_hash:
            index_lock.acquire()
            (word_index,id_to_recipie) = build_reverse_index() # build new index - takes a while
            with open(saved_dict_location, 'wb') as f:
                pickle.dump((word_index,id_to_recipie,current_recipes_hash), f) # save index to file
            recipes_hash = current_recipes_hash

            index_lock.release()
            reload_index.set() # tell search loop that it needs to reload
        check_index.clear() # reset the check index since we have just checked
    
    

def main():
    reload_index = threading.Event() # Flag when search_loop needs to reload index (and when it can begin on the inital run)
    check_index = threading.Event() # Flag for search_loop to tell index_loop to check if it needs to rebuild the index
    index_lock = threading.Lock() # A lock for writing to the index file

    reload_index.clear()
    check_index.set()

    search = threading.Thread(target=search_loop,args=(reload_index,check_index,index_lock),name='search')
    index = threading.Thread(target=index_loop,args=(reload_index,check_index,index_lock),name='index')
    index.start()
    search.start()

main()