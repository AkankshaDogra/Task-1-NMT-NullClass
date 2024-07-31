import numpy as np
from math import log

class NMTModel:
    def __init__(self):
        
        self.vocab_size = 5
        self.hidden_size = 10
        self.output_size = 5
        
        
        self.weights = {
            'encoder': np.random.randn(self.hidden_size, self.vocab_size),
            'decoder': np.random.randn(self.output_size, self.hidden_size)
        }

    def translate(self, input_vector):
        
        encoded_vector = np.dot(self.weights['encoder'], input_vector)
        
        decoded_vector = np.dot(self.weights['decoder'], encoded_vector)
        return decoded_vector

def beam_search_decoder(model, input_sequence, beam_width):
    
    sequences = [[[], 0.0]]
    
    
    for input_step in input_sequence:
        all_candidates = []

        
        for seq, score in sequences:
            
            decoded_output = model.translate(input_step)
            
            
            for i in range(model.output_size):
                new_seq = seq + [i]
                probability = decoded_output[i]
                if probability > 0:
                    new_score = score - log(probability)
                    all_candidates.append((new_seq, new_score))
        
        
        ordered_candidates = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered_candidates[:beam_width]
    
    return sequences

def get_user_input(vocab_size):
    while True:
        try:
            
            user_input = input(f"Enter {vocab_size} probabilities separated by spaces (e.g., 0.1 0.2 0.3 0.4 0.5): \n")
            probabilities = [float(x) for x in user_input.split()]
            
            
            if len(probabilities) == vocab_size:
                return np.array([probabilities])
            else:
                print(f"Please enter exactly {vocab_size} probabilities.")
        except ValueError:
            print("Invalid input. Please enter numeric values separated by spaces.")


model = NMTModel()


vocab_size = model.vocab_size
input_sequence = get_user_input(vocab_size)


beam_width = 3


decoded_sequences = beam_search_decoder(model, input_sequence, beam_width)


for seq, score in decoded_sequences:
    print(f"Sequence: {seq}, Score: {score}")
