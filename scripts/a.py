import pickle

# Path to your pickle file
file_path = "./BN-UNIFIED-NO-SINGLE.pickle"

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)
# c=0   
for writer_id, samples in data['train'].items():
    print(writer_id)
    print("sample count")
    print(len(samples))
    

# Print the content
# print(c)
