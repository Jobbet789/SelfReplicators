import numpy as np
import pygame
import random
import matplotlib.pyplot as plt

# Brainfuck Interpreter
class Brainfuck:
    def __init__(self, max_steps=10000, tape_length=30000):
        self.max_steps = max_steps
        self.tape_length = tape_length

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
                output.append(tape[ptr])  # Output cell
            elif cmd == ',':
                pass  # Input ignore for now
            elif cmd == '[':
                if tape[ptr] == 0:  # Jump forward if zero
                    depth = 1
                    while depth > 0 and pc < len(code) - 1:
                        pc += 1
                        if code[pc] == '[':
                            depth += 1
                        elif code[pc] == ']':
                            depth -= 1
            elif cmd == ']':
                if tape[ptr] != 0:  # Jump back if non-zero
                    depth = 1
                    while depth > 0 and pc > 0:
                        pc -= 1
                        if code[pc] == ']':
                            depth += 1
                        elif code[pc] == '[':
                            depth -= 1
            pc += 1
            steps += 1
        
        return tape[:20]  # Return first 20 cells for detection

    def generate_random_code(self, length):
        commands = ['<', '>', '[', ']', '+', '-', '.', ',']
        return ''.join(random.choice(commands) for _ in range(length))


# Soup Simulation with Pygame Visualization
class Soup:
    def __init__(self, num_strings=100, code_length=20, grid_size=10):
        self.bf = Brainfuck()
        self.strings = [self.bf.generate_random_code(code_length) for _ in range(num_strings)]
        self.entropies = []
        self.replicators_list = []
        self.grid_size = grid_size  # e.g., 10x10 grid for 100 strings
        self.cell_size = 50  # Size of each cell in pixels
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size + 100  # Extra space for entropy bar/text

    def calculate_entropy(self):
        counts = {}
        for s in self.strings:
            counts[s] = counts.get(s, 0) + 1
        probs = [c / len(self.strings) for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def count_replicators(self):
        return sum(1 for s in self.strings 
                   if len(s) * 2 <= 20 and self.bf.run(s)[0:len(s)] == self.bf.run(s)[len(s):2*len(s)])

    def iteration(self):
        idx1, idx2 = random.sample(range(len(self.strings)), 2)
        code1, code2 = self.strings[idx1], self.strings[idx2]
        
        combined = code1 + code2
        tape_after = self.bf.run(combined)
        new_snip = ''.join(chr(x % len('+-<>[].,')) for x in tape_after[:len(code1)] if x != 0)
        if new_snip:
            code1 = code1[:len(code1)//2] + new_snip
        
        self.strings[idx1] = code1[:len(code1)]  # Keep length constant
        self.strings[idx2] = code2
        
        for idx in [idx1, idx2]:
            if random.random() < 0.2:
                s_list = list(self.strings[idx])
                pos = random.randint(0, len(s_list)-1)
                s_list[pos] = random.choice('+-<>[].,')
                self.strings[idx] = ''.join(s_list)

    def visualize(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Brainfuck Soup Visualization")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        
        running = True
        iteration_count = 0
        max_iterations = 10000  # Stop after this or run indefinitely
        
        while running and iteration_count < max_iterations:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Run simulation step
            self.iteration()
            entropy = self.calculate_entropy()
            replicators = self.count_replicators()
            self.entropies.append(entropy)
            self.replicators_list.append(replicators)
            iteration_count += 1
            
            # Draw grid
            screen.fill((0, 0, 0))  # Black background
            for i in range(len(self.strings)):
                x = (i % self.grid_size) * self.cell_size
                y = (i // self.grid_size) * self.cell_size
                color = (0, 255, 0) if (len(self.strings[i]) * 2 <= 20 and self.bf.run(self.strings[i])[0:len(self.strings[i])] == self.bf.run(self.strings[i])[len(self.strings[i]):2*len(self.strings[i])]) else (255, 0, 0)  # Green for replicator, red for normal
                pygame.draw.rect(screen, color, (x, y, self.cell_size, self.cell_size))
                
                # Optional: Draw string length or hash as text
                text = font.render(str(len(self.strings[i])), True, (255, 255, 255))
                screen.blit(text, (x + 10, y + 10))
            
            # Draw entropy bar and text at bottom
            bar_width = int((self.width * (1 - entropy / np.log2(len(self.strings)))) if len(self.strings) > 1 else 0)
            pygame.draw.rect(screen, (0, 0, 255), (0, self.height - 100, bar_width, 50))  # Blue bar for normalized entropy (lower = shorter bar)
            text_entropy = font.render(f"Entropy: {entropy:.2f}", True, (255, 255, 255))
            screen.blit(text_entropy, (10, self.height - 90))
            
            text_replicators = font.render(f"Replicators: {replicators}", True, (255, 255, 255))
            screen.blit(text_replicators, (10, self.height - 60))
            
            text_iteration = font.render(f"Iteration: {iteration_count}", True, (255, 255, 255))
            screen.blit(text_iteration, (10, self.height - 30))
            
            pygame.display.flip()
            clock.tick(10)  # 10 iterations per second for visible changes
        
        pygame.quit()
        # After closing, plot graphs as before
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.entropies)
        plt.xlabel('Iteration')
        plt.ylabel('Entropy')
        plt.title('Entropy over Time')

        plt.subplot(1, 2, 2)
        plt.plot(self.replicators_list)
        plt.xlabel('Iteration')
        plt.ylabel('Number of Replicators')
        plt.title('Replicators over Time')
        plt.show()


if __name__ == '__main__':
    soup = Soup(num_strings=100, code_length=30, grid_size=10)  # 10x10 grid for 100 strings
    soup.visualize()