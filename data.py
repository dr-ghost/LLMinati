import random
import pickle

# Function to generate random dataset
def generate_dataset(num_spaces, num_objects, max_capacity=15, max_object_length=15):
    # Generate random capacities for the spaces
    spaces = [random.randint(5, max_capacity) for _ in range(num_spaces)]
    
    # Generate random lengths for the objects
    objects = [random.randint(1, max_object_length) for _ in range(num_objects)]
    
    return spaces, objects

# Function to generate prompt in the specified format
def generate_prompts(number_of_prompts, num_spaces, num_objects):
    # Generate random spaces and objects
    prompts = []
    for _ in range(number_of_prompts):
        spaces, objects = generate_dataset(num_spaces, num_objects)
        prompt = f"""Given length of spaces with capacities {spaces} and objects with lengths {objects}, assign each object to exactly one space such that the total length of objects in each space does not exceed its capacity, and each object is placed into one space. Output the assignment by first writing OUTPUT followed by S1 - [Oi,Oj,Ok] , S2 - [Om,On]"""
        prompts.append(prompt)
    
    return prompts

# Generate a list of prompts with random spaces and objects
num_spaces = 3
num_objects = 5
number_of_prompts = 5
prompts = generate_prompts(number_of_prompts, num_spaces, num_objects)

# Save the prompts to a pickle file
pickle_filename = 'generated_prompt.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump(prompts, f)

