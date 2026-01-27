import numpy as np
import matplotlib.pyplot as plt
import random


class Brainfuck:
    def __init__(self, max_steps=10000, tape_length=30000):
        self.max_steps = max_steps
        self.tape_length = tape_length
        self.reset()
    
    def reset(self):
        pass

    def run(self, code):
        tape = [0] * self.tape_length  # Memory tape
        ptr = 0  # Pointer
        pc = 0  # Program counter
        steps = 0
        output = []

        while pc < len(code) and steps < self.max_steps:
            cmd = code[pc]
            if cmd == '+':
                tape[ptr] = (tape[ptr] + 1) % 256
            elif cmd == '-':
                tape[ptr] = (tape[ptr] - 1) % 256
            elif cmd == '>':
                ptr = (ptr + 1) % self.tape_length
            elif cmd == '<':
                ptr = (ptr - 1) % self.tape_length
            elif cmd == '.':
                output.append(tape[ptr])  # Output cell (voor toekomst)
            elif cmd == ',':
                pass  # Input negeren voor nu
            elif cmd == '[':
                if tape[ptr] == 0:  # Fix: jump forward if zero
                    depth = 1
                    while depth > 0 and pc < len(code) - 1:
                        pc += 1
                        if code[pc] == '[':
                            depth += 1
                        elif code[pc] == ']':
                            depth -= 1
            elif cmd == ']':
                if tape[ptr] != 0:  # Fix: jump back if non-zero
                    depth = 1
                    while depth > 0 and pc > 0:
                        pc -= 1
                        if code[pc] == ']':
                            depth += 1
                        elif code[pc] == '[':
                            depth -= 1
            pc += 1
            steps += 1
        
        # Return na de loop! (fix indentatie)
        return tape[:20]  # Eerste 20 cellen (verhoog voor langere codes)

    def generate_random_code(self, length):
        commands = ['<', '>', '[', ']', '+', '-', '.', ',']  # Toegevoegd . en , voor meer variatie
        return ''.join(random.choice(commands) for _ in range(length))


class Soup:
    def __init__(self, num_strings=100, code_length=20):
        self.bf = Brainfuck()
        self.strings = [self.bf.generate_random_code(code_length) for _ in range(num_strings)]
        self.entropies = []
        self.replicators_list = []  # Track per iteratie
    
    def run_soup(self, iterations=1000):
        for i in range(iterations):
            self.iteration()

            # Detect replicators (verbeterd: check duplicatie op tape)
            replicators = sum(1 for s in self.strings 
                              if len(s) * 2 <= 20 and self.bf.run(s)[0:len(s)] == self.bf.run(s)[len(s):2*len(s)])

            self.replicators_list.append(replicators)

            # Calculate entropy
            counts = {}
            for s in self.strings:
                counts[s] = counts.get(s, 0) + 1
            probs = [c / len(self.strings) for c in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            self.entropies.append(entropy)

        return self.entropies, self.replicators_list
    
    def iteration(self):
        idx1, idx2 = random.sample(range(len(self.strings)), 2)
        code1, code2 = self.strings[idx1], self.strings[idx2]
        
        # Interact: concat, run, en gebruik output om te muteren (simpele self-mod)
        combined = code1 + code2
        tape_after = self.bf.run(combined)
        # Simpele "modificatie": neem eerste deel van tape als nieuwe code-snip
        new_snip = ''.join(chr(x % len('+-<>[].,')) for x in tape_after[:len(code1)] if x != 0)
        if new_snip:
            code1 = code1[:len(code1)//2] + new_snip  # Vervang deel met tape-invloeden
        
        self.strings[idx1] = code1[:len(code1)]  # Hou lengte constant
        self.strings[idx2] = code2
        
        # Mutatie op beide (verhoog kans)
        for idx in [idx1, idx2]:
            if random.random() < 0.2:  # Verhoogd naar 20%
                s_list = list(self.strings[idx])
                pos = random.randint(0, len(s_list)-1)
                s_list[pos] = random.choice('+-<>[].,')
                self.strings[idx] = ''.join(s_list)


if __name__ == '__main__':
    soup = Soup(num_strings=1000, code_length=30)
    entropies, replicators_list = soup.run_soup(iterations=10_000)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(entropies)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Entropy over Time')

    plt.subplot(1, 2, 2)
    plt.plot(replicators_list)
    plt.xlabel('Iteration')
    plt.ylabel('Number of Replicators')
    plt.title('Replicators over Time')
    plt.show()

    avg_replicators = np.mean(replicators_list)
    print(f"Average number of replicators: {avg_replicators}")
    print(f"Max replicators: {max(replicators_list)}")