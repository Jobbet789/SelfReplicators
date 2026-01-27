import numpy as np 

class Grid:

    def __init__(self, height: int, width: int, density: float = 0.2):
        # Init the grid

        # height: Height of the grid (rows)
        # width: Width of the grid (columns)
        # density: P(0-1) that cell is filled with random token

        self.height = height
        self.width = width
        self.density = density
        self.tokens = ['+', '-', '<', '>', '.', ',', '[', ']']  # Brainfuck commands
        self.grid = self._init_grid()

    def _init_grid(self) -> np.ndarray:
        # Private to init grid

        grid = np.empty((self.height, self.width), dtype='U1') # Unicode string of length 1
        mask = np.random.random((self.height, self.width)) < self.density
        random_tokens = np.random.choice(self.tokens, size=(self.height, self.width))
        grid[mask] = random_tokens[mask]
        grid[~mask] = ' '
        return grid

    def get_cell(self, x: int, y: int) -> str:
        # Value at a cell

        return self.grid[x % self.height, y % self.width]

    def set_cell(self, x: int, y: int, value: str):
        # Set value at a cell

        self.grid[x % self.height, y % self.width] = value

    def __str__(self) -> str:
        # debug/print
        return '\n'.join(''.join(row) for row in self.grid)