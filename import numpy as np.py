import numpy as np
from math import log

class NMTModel:
    def __init__(self):
        self.size_of_vocab = 5
        self.hidden_size = 10
        self.output_size = 5
        self.weights = {
            'encoder': np.random.randn(self.hidden_size, self.size_of_vocab),
            'decoder': np.random.randn(self.output_size, self.hidden_size)
        }

    def translate(self, input_sequence):
        encoded_output = np.dot(self.weights['encoder'], input_sequence)
        decoded_output = np.dot(self.weights['decoder'], encoded_output)
        return decoded_output

def beam_search_decoder_nmt(model, input_sequence, beam_width):
    sequences = [[[], 0.0]]
    
    for input_step in input_sequence:
        all_candidates = []

        for seq, score in sequences:
            decoded_output = model.translate(input_step)
            for j in range(model.output_size):
                candidate_seq = seq + [j]
                probability = decoded_output[j]
                if probability > 0:
                    candidate_score = score - log(probability)
                    all_candidates.append((candidate_seq, candidate_score))
        
        ordered_candidates = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered_candidates[:beam_width]
    
    return sequences

def get_user_input_sequence(size_of_vocab):
    while True:
        try:
            input_str = input(f"Enter probabilities for {size_of_vocab} elements separated by space (e.g., 0.1 0.2 0.3 0.4 0.5): \n")
            probabilities = [float(prob) for prob in input_str.split()]
            if len(probabilities) == size_of_vocab:
                return np.array([probabilities])
            else:
                print(f"Please enter exactly {size_of_vocab} probabilities.")
        except ValueError:
            print("Invalid input. Please enter valid numbers separated by spaces.")


nmt_model = NMTModel()


vocab_size = nmt_model.size_of_vocab
user_input_sequence = get_user_input_sequence(vocab_size)


beam_width = 3


result = beam_search_decoder_nmt(nmt_model, user_input_sequence, beam_width)


for seq, score in result:
    print("Sequence:", seq, "Score:", score)
