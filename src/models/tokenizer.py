# tokenizer.py
import re
from tqdm import tqdm

class BengaliGraphemeTokenizer:
    def __init__(self, output_max_len=95):
        self.vocab = {}
        self.index2word = {}
        self.output_max_len = output_max_len
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.special_tokens = [self.pad_token, self.unk_token]

    def split_graphemes(self, text):
        """
        Splits Bengali text into grapheme clusters including complex conjuncts.
        """
        if not text:
            return []

        graphemes = []
        chars = list(text)
        current_cluster = ""
        
        i = 0
        while i < len(chars):
            char = chars[i]
            current_cluster += char
            
            while i + 1 < len(chars):
                next_char = chars[i+1]
                
                # Logic: Grab next char if it is a modifier OR if we are in a virama chain
                is_modifier = ('\u0981' <= next_char <= '\u0983') or \
                              (next_char == '\u09BC') or \
                              ('\u09BE' <= next_char <= '\u09CC') or \
                              ('\u09D7' == next_char) or \
                              ('\u09E2' <= next_char <= '\u09E3')

                # Current char is Virama (Linker)
                is_virama_connector = (char == '\u09CD')
                
                # Next char is Virama (Linker)
                next_is_virama = (next_char == '\u09CD')

                if is_modifier or is_virama_connector or next_is_virama:
                    current_cluster += next_char
                    i += 1
                    char = next_char 
                else:
                    break
            
            graphemes.append(current_cluster)
            current_cluster = ""
            i += 1
            
        return graphemes

    def build_vocab(self, dataset_texts):
        """Scans the entire dataset to build the grapheme vocabulary"""
        unique_graphemes = set()
        print("🔨 Building Grapheme Vocabulary...")
        
        for text in tqdm(dataset_texts, desc="Scanning graphemes"):
            graphemes = self.split_graphemes(text)
            unique_graphemes.update(graphemes)
            
        sorted_graphemes = sorted(list(unique_graphemes))
        
        # Add special tokens
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        start_idx = len(self.special_tokens)
        
        for i, g in enumerate(sorted_graphemes):
            self.vocab[g] = start_idx + i
            
        self.index2word = {v: k for k, v in self.vocab.items()}
        print(f"✅ Vocab built. Size: {len(self.vocab)} unique graphemes.")

    def encode(self, text):
        graphemes = self.split_graphemes(text)
        indices = [self.vocab.get(g, self.vocab[self.unk_token]) for g in graphemes]
        
        pad_idx = self.vocab[self.pad_token]
        if len(indices) < self.output_max_len:
            indices += [pad_idx] * (self.output_max_len - len(indices))
        else:
            indices = indices[:self.output_max_len]
            
        return indices

    def get_vocab_size(self):
        return len(self.vocab)