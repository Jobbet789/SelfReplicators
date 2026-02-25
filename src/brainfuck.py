"""
Brainfuck interpreter for emergent self-replication simulation.
"""

import numpy as np
from typing import Optional, Tuple


class BrainfuckInterpreter:
    """
    Executes Brainfuck code on a shared tape with read/write heads.
    Supports local interactions for emergent replication.
    """
    
    def __init__(self, tape_size: int = 1000, max_steps: int = 1000):
        """
        Initialize the Brainfuck interpreter.
        
        Args:
            tape_size: Size of the memory tape
            max_steps: Maximum execution steps to prevent infinite loops
        """
        self.tape_size = tape_size
        self.max_steps = max_steps
        self.tape = np.zeros(tape_size, dtype=np.int32)
        self.pointer = 0
        self.steps_executed = 0
        
    def reset(self):
        """Reset the interpreter state."""
        self.tape = np.zeros(self.tape_size, dtype=np.int32)
        self.pointer = 0
        self.steps_executed = 0
        
    def execute(self, code: str, start_pointer: int = 0, 
                output_buffer: Optional[list] = None,
                input_buffer: Optional[list] = None) -> Tuple[bool, int]:
        """
        Execute Brainfuck code on the tape.
        
        Args:
            code: Brainfuck program string
            start_pointer: Starting position on the tape
            output_buffer: List to collect output (for '.' command)
            input_buffer: List of input values (for ',' command)
            
        Returns:
            Tuple of (success: bool, steps_taken: int)
        """
        if output_buffer is None:
            output_buffer = []
        if input_buffer is None:
            input_buffer = []
            
        self.pointer = start_pointer % self.tape_size
        code_pointer = 0
        self.steps_executed = 0
        input_index = 0
        
        # Build jump table for brackets
        jump_table = self._build_jump_table(code)
        if jump_table is None:
            return False, 0  # Unmatched brackets
        
        while code_pointer < len(code) and self.steps_executed < self.max_steps:
            command = code[code_pointer]
            self.steps_executed += 1
            
            if command == '+':
                # Increment the byte at the pointer
                self.tape[self.pointer] = (self.tape[self.pointer] + 1) % 256
                
            elif command == '-':
                # Decrement the byte at the pointer
                self.tape[self.pointer] = (self.tape[self.pointer] - 1) % 256
                
            elif command == '>':
                # Move pointer right (circular)
                self.pointer = (self.pointer + 1) % self.tape_size
                
            elif command == '<':
                # Move pointer left (circular)
                self.pointer = (self.pointer - 1) % self.tape_size
                
            elif command == '.':
                # Output the byte at the pointer
                output_buffer.append(self.tape[self.pointer])
                
            elif command == ',':
                # Input a byte to the pointer
                if input_index < len(input_buffer):
                    self.tape[self.pointer] = input_buffer[input_index] % 256
                    input_index += 1
                else:
                    self.tape[self.pointer] = 0  # Default to 0 if no input
                    
            elif command == '[':
                # Jump forward if current cell is 0
                if self.tape[self.pointer] == 0:
                    code_pointer = jump_table[code_pointer]
                    
            elif command == ']':
                # Jump backward if current cell is not 0
                if self.tape[self.pointer] != 0:
                    code_pointer = jump_table[code_pointer]
            
            code_pointer += 1
        
        success = self.steps_executed < self.max_steps
        return success, self.steps_executed
    
    def _build_jump_table(self, code: str) -> Optional[dict]:
        """
        Build a jump table for matching brackets.
        
        Args:
            code: Brainfuck program string
            
        Returns:
            Dictionary mapping bracket positions, or None if unmatched
        """
        jump_table = {}
        stack = []
        
        for i, char in enumerate(code):
            if char == '[':
                stack.append(i)
            elif char == ']':
                if not stack:
                    return None  # Unmatched closing bracket
                open_bracket = stack.pop()
                jump_table[open_bracket] = i
                jump_table[i] = open_bracket
        
        if stack:
            return None  # Unmatched opening bracket
            
        return jump_table
    
    def get_tape_segment(self, start: int, length: int) -> np.ndarray:
        """
        Get a segment of the tape (circular).
        
        Args:
            start: Starting position
            length: Number of cells to retrieve
            
        Returns:
            Numpy array of tape values
        """
        indices = [(start + i) % self.tape_size for i in range(length)]
        return self.tape[indices]
    
    def set_tape_segment(self, start: int, values: np.ndarray):
        """
        Set a segment of the tape (circular).
        
        Args:
            start: Starting position
            values: Array of values to write
        """
        for i, val in enumerate(values):
            self.tape[(start + i) % self.tape_size] = val % 256


class GridBrainfuckEngine:
    """
    Executes Brainfuck programs from a 2D grid with local interactions.
    This is the core engine for emergent self-replication.
    """
    
    def __init__(self, grid, interaction_radius: int = 1, 
                 max_program_length: int = 64, max_steps: int = 100):
        """
        Initialize the grid-based Brainfuck execution engine.
        
        Args:
            grid: Grid instance containing the code
            interaction_radius: How far programs can read/write (cells)
            max_program_length: Maximum length of programs to execute
            max_steps: Maximum steps per program execution
        """
        self.grid = grid
        self.interaction_radius = interaction_radius
        self.max_program_length = max_program_length
        self.max_steps = max_steps
        
        # Create a shared tape representing the grid
        self.tape_size = grid.height * grid.width
        self.interpreter = BrainfuckInterpreter(
            tape_size=self.tape_size, 
            max_steps=max_steps
        )
        
    def _grid_to_tape_index(self, x: int, y: int) -> int:
        """Convert 2D grid coordinates to 1D tape index."""
        return (x % self.grid.height) * self.grid.width + (y % self.grid.width)
    
    def _tape_to_grid_coords(self, index: int) -> Tuple[int, int]:
        """Convert 1D tape index to 2D grid coordinates."""
        index = index % self.tape_size
        x = index // self.grid.width
        y = index % self.grid.width
        return x, y
    
    def sync_grid_to_tape(self):
        """Synchronize grid contents to the interpreter's tape."""
        for x in range(self.grid.height):
            for y in range(self.grid.width):
                token = self.grid.get_cell(x, y)
                tape_index = self._grid_to_tape_index(x, y)
                if token in self.grid.tokens:
                    # Map token to ASCII value
                    self.interpreter.tape[tape_index] = ord(token)
                else:
                    self.interpreter.tape[tape_index] = 0
    
    def sync_tape_to_grid(self):
        """Synchronize interpreter's tape back to grid."""
        for x in range(self.grid.height):
            for y in range(self.grid.width):
                tape_index = self._grid_to_tape_index(x, y)
                value = self.interpreter.tape[tape_index]
                # Convert back to character if it's a valid Brainfuck token
                char = chr(value) if value > 0 and value < 128 else ' '
                if char in self.grid.tokens:
                    self.grid.set_cell(x, y, char)
                else:
                    self.grid.set_cell(x, y, ' ')
    
    def extract_program(self, x: int, y: int, length: int) -> str:
        """
        Extract a program from the grid starting at (x, y).
        
        Args:
            x, y: Starting coordinates
            length: Length of program to extract
            
        Returns:
            Program string
        """
        program = []
        for i in range(length):
            # Read horizontally, wrapping around
            cell_x = x
            cell_y = (y + i) % self.grid.width
            token = self.grid.get_cell(cell_x, cell_y)
            program.append(token if token in self.grid.tokens else ' ')
        return ''.join(program).strip()
    
    def execute_at_position(self, x: int, y: int) -> bool:
        """
        Execute a program starting at grid position (x, y).
        
        Args:
            x, y: Starting position
            
        Returns:
            True if execution was successful
        """
        # Extract program
        program = self.extract_program(x, y, self.max_program_length)
        
        if not program:
            return False
        
        # Sync grid to tape
        self.sync_grid_to_tape()
        
        # Execute
        tape_index = self._grid_to_tape_index(x, y)
        success, steps = self.interpreter.execute(
            program, 
            start_pointer=tape_index
        )
        
        # Sync tape back to grid
        if success:
            self.sync_tape_to_grid()
        
        return success
