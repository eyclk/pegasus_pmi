from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
from rouge_score import rouge_scorer
import re
import numpy as np
from itertools import combinations
import itertools

def conditional_probability(target_sentence, context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt').to(device)

    # Tokenize the target sentence
    target_inputs = tokenizer(target_sentence, return_tensors='pt').to(device)

    losses = model(context_inputs['input_ids'], labels=target_inputs['input_ids'])
    #print("Probability of target sentence given context sentence:", losses.loss.item())

    return math.exp(-losses.loss.item())

def marginal_probability(context_sentence):
    # Tokenize the context sentence
    context_inputs = tokenizer(context_sentence, return_tensors='pt').to(device)

    # Generate output
    output = model.generate(context_inputs['input_ids'], max_new_tokens=40, num_beams=5, no_repeat_ngram_size=2,
                            early_stopping=True)

    losses = model(context_inputs['input_ids'], labels=output)
    # Print probability
    #print("Probability of target sentence given context sentence:", p_x_given_y)

    return math.exp(-losses.loss.item())

def get_npmi_matrix(sentences, method=1, batch_size=1):
    """
    Accepts a list of sentences of length n and returns 3 objects:
    - Normalised PMI nxn matrix - temp
    - PMI nxn matrix - temp2
    - List of length n indicating sentence-wise surprisal i.e. p(sentence) - p

    To optimize performance, we do the forward pass batchwise by assembling the batch and maintaining batch indices
    For each batch we call get_probabilities
    """
    temp = np.zeros((len(sentences), len(sentences)))
    temp2 = np.zeros((len(sentences), len(sentences)))
    batch_indices = {}
    batch = []
    batchCount = 0
    batchSize = batch_size
    # print(len(sentences))
    c = 0
    p = []
    for i in range(len(sentences)):
        result = marginal_probability([sentences[i]])
        p.append(result)
        #try:
        #    p.append(sum([math.log(i) for i in result[0]]))
        #except:
        #    print("Math domain error surprise", i)
        #    return temp, temp2, p
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                temp[i][j] = -1
                temp2[i][j] = -1
                continue
            article = sentences[i] + " " + sentences[j]
            # print(article)
            batch_indices[str(i) + "-" + str(j) + "-" + str(len(sentences[i].split()))] = batchCount
            batch.append(article)
            batchCount += 1

            if batchCount == batchSize or (i == len(sentences) - 1 and j == len(sentences) - 1):
                # print(batch)
                c += 1
                #result = marginal_probability(batch)
                for key in batch_indices.keys():
                    # print(key)
                    # print(key.split("-"))
                    idx_i, idx_j, idx_l = [int(idx) for idx in key.split("-")]
                    try:
                        p_x_given_y = conditional_probability(batch[idx_j], batch[idx_i])
                        #pxy = sum([math.log(q) for q in result[batch_indices[key]][idx_l:]])
                        py = p[idx_j]
                        px = p[idx_i]
                        # Normalized Mutual Information H(Y)-H(Y|C)
                        temp[idx_i][idx_j] = py - p_x_given_y
                        #temp[idx_i][idx_j] = (pxy - py) / (-1 * (pxy + px))
                        temp2[idx_i][idx_j] = math.log2(p_x_given_y / px)
                        #temp2[idx_i][idx_j] = (pxy - py)
                    except ZeroDivisionError:
                        print("Zero division error ", idx_i, idx_j)
                        temp[idx_i][idx_j] = -1
                        temp2[idx_i][idx_j] = -1
                    except:
                        print("Math Domain Error", i, j)
                    if temp[idx_i][idx_j] > 1 or temp[idx_i][idx_j] < -1:
                        print("Normalise assert ", temp[idx_i][idx_j], idx_i, idx_j)
                batchCount = 0
                batch = []
                batch_indices = {}
    return temp, temp2, p

def calculate_pmi(p_x_given_y, p_x):
    """
    Calculate pointwise mutual information (PMI) between two random variables X and Y.

    Parameters:
    p_x (float): Probability of X
    p_y (float): Probability of Y
    p_xy (float): Joint probability of X and Y

    Returns:
    pmi (float): Pointwise mutual information between X and Y
    """
    pmi = math.log2(p_x_given_y / p_x)
    return pmi

# Load pre-trained model tokenizer (vocabulary)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load pre-trained model (weights)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Context sentence
#context_sentence = "I love to read books."
#context_sentence = "Apple's lawyers said they were willing to publish a clarification."
context_sentence = "Michelle, of South Shields, Tyneside, says she feels like a new woman after dropping from dress size 30 to size 12."

# Target sentence
#target_sentence = "I also enjoy reading articles."
#target_sentence = "However the company does not accept that it misled customers."
target_sentence = "Michelle weighed 25st 3lbs when she joined the group in April 2013 and has since dropped to 12st " \
                  "10lbs."

p_x_given_y = conditional_probability(target_sentence,context_sentence)
p_x = marginal_probability(context_sentence)
pmi = calculate_pmi(p_x_given_y, p_x)
print(pmi)

def split_into_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences

doc1 = "US Treasury Secretary Robert Rubin arrived in Malaysia Sunday for a two-day visit to discuss the " \
            "regional economic situation, the US Embassy said. Rubin, on a tour of Asia's economic trouble spots, " \
            "arrived from Beijing, where he had accompanied US President Bill Clinton on a visit. Rubin was " \
            "scheduled to meet and have dinner with Finance Minister Anwar Ibrahim on Sunday. On Monday, Rubin will " \
            "meet privately with Prime Minister Mahathir Mohamad and separately with senior Malaysian and American " \
            "business leaders, the embassy said in a statement. Rubin will leave Monday for Thailand and South Korea."
#
"""file_path = 'concatenated_d30001t.txt'

with open(file_path, 'r') as file:
    doc = file.read()"""

sentences = split_into_sentences(doc1)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
#similarities = []
scores_with_contents = []

normalised, vanilla, surprise = get_npmi_matrix(sentences, batch_size = 10)

upper_triangle = vanilla[np.triu_indices(len(sentences), k = 1)]
print(upper_triangle)

lower_triangle = vanilla[np.tril_indices(len(sentences), k = -1)]
print(lower_triangle)

def upper_tri_diagonal_iter(matrix):
    result = []
    for diag in range(1, len(matrix)):
        for row in range(0, len(matrix)-diag):
            col = row + diag
            result.append([sentences[row], sentences[col], vanilla[row][col]])
    return result

def lower_tri_diagonal_iter(matrix):
    result = []
    for diag in range(1, len(matrix)):
        for row in range(0, len(matrix)-diag):
            col = row + diag
            result.append([sentences[col], sentences[row], vanilla[col][row]])
    return result

upper_test = upper_tri_diagonal_iter(vanilla)
#print(test)

lower_test = lower_tri_diagonal_iter(vanilla)
#print(lower_test)

#for el in test:
#    print(el)

#print(sorted(test, key = lambda x: x[2], reverse = True))
#print(sorted(lower_test, key = lambda x: x[2], reverse = True))
print(sorted(upper_test, key = lambda x: x[2], reverse = True))


for i in sentences:
    #similarities_i = []
    for j in sentences:
        if i!= j:
            scores = scorer.score(i, j)
            #similarities.append((scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure +
                                     #scores['rougeLsum'].fmeasure) / 4)
            scores_with_contents.append([i, j, scores['rougeLsum'].fmeasure])
            #pmi_score = calculate_pmi(conditional_probability(j,i),marginal_probability(i))
            #scores_with_contents.append([i + "\n", j + "\n", pmi_score])
            #print(i, j, pmi_score)
    #print(sorted(scores_with_contents, key = lambda x: x[2], reverse = True)[1])
print(sorted(scores_with_contents, key = lambda x: x[2], reverse = True))

