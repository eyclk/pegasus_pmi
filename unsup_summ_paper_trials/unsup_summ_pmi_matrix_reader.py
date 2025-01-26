import pickle


data = pickle.load(open("./sent_test_set_0.pkl", "rb"))
doc_number = 0
print("----------->  ", data[doc_number]['vanilla'].shape, "\n")
print(data[doc_number]['vanilla'])
