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
        tape = [0] * self.tape_length
        ptr = 0
        pc = 0
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
                output.append(tape[ptr])
            elif cmd == ',':
                pass
            elif cmd == '[':
                if tape[ptr] == 0:
                    depth = 1
                    while depth > 0 and pc < len(code) - 1:
                        pc += 1
                        if code[pc] == '[':
                            depth += 1
                        elif code[pc] == ']':
                            depth -= 1
            elif cmd == ']':
                if tape[ptr] != 0:
                    depth = 1
                    while depth > 0 and pc > 0:
                        pc -= 1
                        if code[pc] == ']':
                            depth += 1
                        elif code[pc] == '[':
                            depth -= 1
            pc += 1
            steps += 1
        
        return tape[:50], output  # Return tape and output for better detection

    def generate_random_code(self, length):
        commands = ['<', '>', '[', ']', '+', '-', '.', ',']
        return ''.join(random.choice(commands) for _ in range(length))


class Soup:
    def __init__(self, num_strings=100, code_length=20, grid_size=10):
        self.bf = Brainfuck()
        self.grid_size = grid_size
        self.strings = [self.bf.generate_random_code(code_length) for _ in range(num_strings)]
        self.grid = np.array(self.strings).reshape(grid_size, grid_size)
        self.entropies = []
        self.replicators_list = []
        self.cell_size = 50
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size + 100
        self.fitness = np.zeros((grid_size, grid_size))  # Track fitness for visualization

    def calculate_entropy(self):
        counts = {}
        for s in self.grid.flatten():
            counts[s] = counts.get(s, 0) + 1
        probs = [c / len(self.grid.flatten()) for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def is_replicator(self, code):
        tape, output = self.bf.run(code)
        # Check 1: Tape duplicatie
        tape_check = len(code) * 2 <= 50 and tape[:len(code)] == tape[len(code):2*len(code)]
        # Check 2: Output lijkt op code
        output_check = output and ''.join(chr(x % 128) for x in output).startswith(code[:5])
        return tape_check or output_check

    def iteration(self):
        # Kies random cel en een buur
        i, j = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        neighbors = [(ni, nj) for ni, nj in neighbors if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size]
        if not neighbors:
            return
        ni, nj = random.choice(neighbors)
        
        code1, code2 = self.grid[i, j], self.grid[ni, nj]
        combined = code1 + code2
        tape_after, _ = self.bf.run(combined)
        
        # Self-modificatie: Gebruik tape om nieuwe code te maken
        new_snip = ''.join(chr(x % len('+-<>[].,')) for x in tape_after[:len(code1)] if x != 0)
        if new_snip and self.is_replicator(code1 + new_snip):
            self.grid[i, j] = (code1 + new_snip)[:len(code1)]
            self.fitness[i, j] += 1  # Verhoog fitness bij replicatie
            # Verspreid naar buur als replicator
            if random.random() < 0.5:
                self.grid[ni, nj] = self.grid[i, j]
                self.fitness[ni, nj] = self.fitness[i, j]
        
        # Mutatie
        for pos in [(i, j), (ni, nj)]:
            if random.random() < 0.3:
                s_list = list(self.grid[pos])
                pos_idx = random.randint(0, len(s_list)-1)
                s_list[pos_idx] = random.choice('+-<>[].,')
                self.grid[pos] = ''.join(s_list)

    def visualize(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Brainfuck Soup Visualization")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        
        running = True
        iteration_count = 0
        max_iterations = 10000
        
        while running and iteration_count < max_iterations:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Klik om code te tonen
                    mx, my = pygame.mouse.get_pos()
                    i, j = my // self.cell_size, mx // self.cell_size
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        print(f"Code at ({i}, {j}): {self.grid[i, j]}")
            
            # Run simulatie
            self.iteration()
            entropy = self.calculate_entropy()
            replicators = sum(1 for s in self.grid.flatten() if self.is_replicator(s))
            self.entropies.append(entropy)
            self.replicators_list.append(replicators)
            iteration_count += 1
            
            # Teken grid
            screen.fill((0, 0, 0))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x, y = j * self.cell_size, i * self.cell_size
                    # Puls-effect voor replicators
                    is_rep = self.is_replicator(self.grid[i, j])
                    brightness = 255 if not is_rep else int(128 + 127 * np.sin(pygame.time.get_ticks() / 1000))
                    color = (0, brightness, 0) if is_rep else (255, 0, 0)
                    pygame.draw.rect(screen, color, (x, y, self.cell_size, self.cell_size))
                    
                    # Toon fitness
                    text = font.render(str(int(self.fitness[i, j])), True, (255, 255, 255))
                    screen.blit(text, (x + 10, y + 10))
            
            # Teken entropy bar en stats
            max_entropy = np.log2(len(self.grid.flatten())) if len(self.grid.flatten()) > 1 else 1
            bar_width = int(self.width * (1 - entropy / max_entropy))
            pygame.draw.rect(screen, (0, 0, 255), (0, self.height - 100, bar_width, 50))
            text_entropy = font.render(f"Entropy: {entropy:.2f}", True, (255, 255, 255))
            screen.blit(text_entropy, (10, self.height - 90))
            text_replicators = font.render(f"Replicators: {replicators}", True, (255, 255, 255))
            screen.blit(text_replicators, (10, self.height - 60))
            text_iteration = font.render(f"Iteration: {iteration_count}", True, (255, 255, 255))
            screen.blit(text_iteration, (10, self.height - 30))
            
            pygame.display.flip()
            clock.tick(5)  # Langzamer voor zichtbaarheid
        
        pygame.quit()
        # Plot grafieken
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
    soup = Soup(num_strings=100, code_length=20, grid_size=10)
    soup.visualize()