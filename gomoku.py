"""
Gomoku Game with AI
Copyright (C) 2024  JSettler@GitHub.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import pygame
import numpy as np
from numba import jit
import time
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 800
BOARD_SIZE = 25
CELL_SIZE = WINDOW_SIZE // BOARD_SIZE
STONE_RADIUS = int(CELL_SIZE * 0.4)

# AI Settings
ABSOLUTE_MAX_DEPTH = 12  # Maximum allowed depth
DEFAULT_DEPTH = 2       # Default depth if no argument provided
DEFAULT_SIDE = "first"  # Default side for the bot

# Colors
BACKGROUND = (50, 50, 50)
GRID_COLOR = (0, 0, 0)
PLAYER_COLOR = (65, 105, 225)  # Royal Blue
BOT_COLOR = (220, 20, 60)      # Crimson
LAST_MOVE_HIGHLIGHT = (80, 80, 80)  # Light grey for last bot move

def print_usage():
    print("Usage: python3 gomoku.py <depth> [side]")
    print("  depth: Search depth in plies (1-12)")
    print("  side: 'first' or 'second' (optional, defaults to 'first')")
    print("Example: python3 gomoku.py 5 second")

def get_parameters():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print_usage()
        print(f"\nNo or invalid parameters provided. Using defaults:")
        print(f"Depth: {DEFAULT_DEPTH}")
        print(f"Side: {DEFAULT_SIDE}")
        return DEFAULT_DEPTH, DEFAULT_SIDE
        
    try:
        # Get depth parameter
        depth = int(sys.argv[1])
        if depth < 1:
            print(f"Depth must be at least 1. Using default depth of {DEFAULT_DEPTH}")
            depth = DEFAULT_DEPTH
        elif depth > ABSOLUTE_MAX_DEPTH:
            print(f"Depth {depth} exceeds maximum allowed depth of {ABSOLUTE_MAX_DEPTH}.")
            print(f"Using maximum allowed depth of {ABSOLUTE_MAX_DEPTH}")
            depth = ABSOLUTE_MAX_DEPTH
        
        # Get side parameter if provided
        side = sys.argv[2].lower() if len(sys.argv) == 3 else DEFAULT_SIDE
        if side not in ["first", "second"]:
            print(f"Invalid side parameter '{side}'. Must be 'first' or 'second'.")
            print(f"Using default side: {DEFAULT_SIDE}")
            side = DEFAULT_SIDE
            
        print(f"Using search depth of {depth} ply")
        print(f"Bot playing: {side}")
        return depth, side
        
    except ValueError:
        print(f"Invalid depth parameter. Using default depth of {DEFAULT_DEPTH}")
        return DEFAULT_DEPTH, DEFAULT_SIDE

# Get parameters at startup
MAX_PLY_DEPTH, BOT_SIDE = get_parameters()
BOT_PLAYS_FIRST = (BOT_SIDE == "first")



# Setup display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('PyGomoku 1.0')


def coord_to_notation(x, y):
    """Convert 0-based coordinates to board notation (A-Y, 1-25)"""
    col = chr(x + ord('A'))
    row = str(BOARD_SIZE - y)  # Invert row number as board counts from bottom
    return f"{col}{row}"

def notation_to_coord(notation):
    """Convert board notation (A-Y, 1-25) to 0-based coordinates"""
    col = ord(notation[0].upper()) - ord('A')
    row = BOARD_SIZE - int(notation[1:])  # Invert row number
    return col, row


@jit(nopython=True)
def check_winner(board, x, y, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        # Check forward
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE) or board[ny][nx] != player:
                break
            count += 1
        # Check backward
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE) or board[ny][nx] != player:
                break
            count += 1
        if count >= 5:
            return True



@jit(nopython=True)
def check_five_in_row(board, player, x, y):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        # Check forward
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE) or board[ny][nx] != player:
                break
            count += 1
        # Check backward
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE) or board[ny][nx] != player:
                break
            count += 1
        if count >= 5:
            return True
    return False

@jit(nopython=True)
def check_open_four(board, player, x, y):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        open_ends = 0
        
        # Check forward
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                if count == 4:
                    open_ends += 1
                break
            else:
                break
                
        # Check backward
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                if count == 4:
                    open_ends += 1
                break
            else:
                break
                
        if count == 4 and open_ends == 2:
            return True
    return False


@jit(nopython=True)
def check_half_open_four_and_open_three(board, player, x, y):
    has_half_open_four = False
    has_open_three = False
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    # Check each direction for half-open four
    for dx, dy in directions:
        count = 1
        blocked_ends = 0
        
        # Check forward
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                blocked_ends += 1
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                break
            else:
                blocked_ends += 1
                break
                
        # Check backward
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                blocked_ends += 1
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                break
            else:
                blocked_ends += 1
                break
                
        if count == 4 and blocked_ends == 1:
            has_half_open_four = True
            
        # Check for open three
        count = 1
        open_ends = 0
        
        # Reset and check forward
        for i in range(1, 4):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                open_ends += 1
                break
            else:
                break
                
        # Check backward
        for i in range(1, 4):
            nx, ny = x - dx * i, y - dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                open_ends += 1
                break
            else:
                break
                
        if count == 3 and open_ends == 2:
            has_open_three = True
            
        if has_half_open_four and has_open_three:
            return True
            
    return False


@jit(nopython=True)
def check_double_three(board, player, x, y):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    open_threes = 0
    
    for dx, dy in directions:
        count = 1
        open_ends = 0
        
        # Check forward
        for i in range(1, 4):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                open_ends += 1
                break
            else:
                break
                
        # Check backward
        for i in range(1, 4):
            nx, ny = x - dx * i, y - dy * i
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                break
            if board[ny][nx] == player:
                count += 1
            elif board[ny][nx] == 0:
                open_ends += 1
                break
            else:
                break
                
        if count == 3 and open_ends == 2:
            open_threes += 1
            
        if open_threes >= 2:
            return True
            
    return False


@jit(nopython=True)
def find_priority_move(board, is_bot):
    player = 1 if (is_bot and BOT_PLAYS_FIRST) or (not is_bot and not BOT_PLAYS_FIRST) else -1
    opponent = -player
    
    # Priority 1: Check for immediate win
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = player
                if check_five_in_row(board, player, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0

    # Priority 2: Block opponent's immediate win
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = opponent
                if check_five_in_row(board, opponent, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0

    # Priority 3: Create open four
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = player
                if check_open_four(board, player, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0
    
    # Priority 4: Block opponent's open four
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = opponent
                if check_open_four(board, opponent, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0

    # Priority 5: Create half-open-four + open-three fork
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = player
                if check_half_open_four_and_open_three(board, player, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0

    # Priority 6: Block opponent's half-open-four + open-three fork
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = opponent
                if check_half_open_four_and_open_three(board, opponent, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0
    
    # Priority 7: Create double three fork
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = player
                if check_double_three(board, player, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0
    
    # Priority 8: Block opponent's double three fork
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == 0:
                board[y][x] = opponent
                if check_double_three(board, opponent, x, y):
                    board[y][x] = 0
                    return (x, y, True)
                board[y][x] = 0
    
    return (-1, -1, False)  # No priority move found


#----------------------------------------------------------------------------------------------------------

@jit(nopython=True)
def evaluate_sequence(sequence, player, is_bot_first):
    score = 0
    length = len(sequence)
    count = 0
    blocks = 0
    empty_before = False
    
    # Adjust player value based on bot's side
    bot_value = 1 if is_bot_first else -1
    multiplier = 1 if player == bot_value else -1
    
    # Count consecutive stones and blocks
    for i in range(length):
        if sequence[i] == player:
            count += 1
        elif sequence[i] == 0:
            if count > 0:
                empty_before = True
            if empty_before and count > 0:
                # Score based on sequence length and openness
                if count >= 5:
                    score += 1000000
                elif count == 4:
                    score += 50000 if blocks == 0 else 10000
                elif count == 3:
                    score += 5000 if blocks == 0 else 1000
                elif count == 2:
                    score += 500 if blocks == 0 else 100
                count = 0
                blocks = 0
                empty_before = True
        else:
            if count > 0:
                blocks += 1
                # Score based on sequence length and blocks
                if count >= 5:
                    score += 1000000
                elif count == 4:
                    score += 50000 if blocks == 0 else 10000
                elif count == 3:
                    score += 5000 if blocks == 0 else 1000
                elif count == 2:
                    score += 500 if blocks == 0 else 100
            count = 0
            blocks += 1
            empty_before = False
            
    # Handle remaining sequence
    if count > 0:
        if count >= 5:
            score += 1000000
        elif count == 4:
            score += 50000 if blocks == 0 else 10000
        elif count == 3:
            score += 5000 if blocks == 0 else 1000
        elif count == 2:
            score += 500 if blocks == 0 else 100
            
    return score * multiplier


@jit(nopython=True)
def evaluate_position(board, x, y, player, is_bot_first):
    score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        sequence = np.zeros(11, dtype=np.int32)
        # Get sequence centered at (x,y)
        for i in range(-5, 6):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                sequence[i + 5] = board[ny][nx]
            else:
                sequence[i + 5] = -player  # Treat out of bounds as blocked
        
        score += evaluate_sequence(sequence, player, is_bot_first)
    
    # Add position-based bonus (always positive for bot's moves)
    center = BOARD_SIZE // 2
    distance_to_center = abs(x - center) + abs(y - center)
    position_bonus = (BOARD_SIZE - distance_to_center) * 10
    bot_value = 1 if is_bot_first else -1
    score += position_bonus if player == bot_value else -position_bonus
    
    return score


@jit(nopython=True)
def evaluate_board(board, is_bot_first):
    score = 0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != 0:
                score += evaluate_position(board, x, y, board[y][x], is_bot_first)
    return score


@jit(nopython=True)
def get_valid_moves(board, radius=2):
    moves = []
    has_stones = False
    
    # Check if there are any stones on the board
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != 0:
                has_stones = True
                break
        if has_stones:
            break
    
    # If board is empty, return center position
    if not has_stones:
        center = BOARD_SIZE // 2
        return [(center, center)]
    
    # Otherwise, look for empty positions near existing stones
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != 0:
                # Check surrounding positions within radius
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
                            board[ny][nx] == 0 and (nx, ny) not in moves):
                            moves.append((nx, ny))
    
    return moves if moves else [(x, y) for x in range(BOARD_SIZE) 
                              for y in range(BOARD_SIZE) if board[y][x] == 0]

@jit(nopython=True)
def minimax(board, depth, alpha, beta, is_maximizing, is_bot_first):
    if depth == 0:
        return evaluate_board(board, is_bot_first)
    
    valid_moves = get_valid_moves(board)
    bot_value = 1 if is_bot_first else -1
    
    if is_maximizing:
        max_eval = -float('inf')
        for x, y in valid_moves:
            if board[y][x] == 0:
                board[y][x] = bot_value
                eval = minimax(board, depth - 1, alpha, beta, False, is_bot_first)
                board[y][x] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float('inf')
        for x, y in valid_moves:
            if board[y][x] == 0:
                board[y][x] = -bot_value
                eval = minimax(board, depth - 1, alpha, beta, True, is_bot_first)
                board[y][x] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval



@jit(nopython=True)
def bot_move(board, is_bot_first, max_depth):  # Added parameters needed for numba
    valid_moves = get_valid_moves(board)
    best_score = -float('inf')
    best_move = (valid_moves[0][0], valid_moves[0][1])  # Explicit tuple for numba
    alpha = -float('inf')
    beta = float('inf')
    bot_value = 1 if is_bot_first else -1
    
    for move in valid_moves:
        x, y = move[0], move[1]
        if board[y][x] == 0:
            board[y][x] = bot_value
            score = minimax(board, max_depth, alpha, beta, False, is_bot_first)
            board[y][x] = 0
            
            if score > best_score:
                best_score = score
                best_move = (x, y)
    
    return best_move[0], best_move[1]


@jit(nopython=True)
def improved_bot_move(board, is_bot_first, max_depth):  # Added parameters needed for numba
    # First check for priority moves
    x, y, found = find_priority_move(board, True)
    if found:
        return x, y
        
    # If no priority move, use regular minimax search
    return bot_move(board, is_bot_first, max_depth)


def draw_board(last_bot_move):
    screen.fill(BACKGROUND)
    
    # Draw highlighted square for last bot move
    if last_bot_move is not None:
        x, y = last_bot_move
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, LAST_MOVE_HIGHLIGHT, rect)
    
    # Draw grid lines
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, GRID_COLOR,
                        (i * CELL_SIZE, 0),
                        (i * CELL_SIZE, WINDOW_SIZE))
        pygame.draw.line(screen, GRID_COLOR,
                        (0, i * CELL_SIZE),
                        (WINDOW_SIZE, i * CELL_SIZE))


def draw_stones(board):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] != 0:
                color = BOT_COLOR if board[y][x] == 1 else PLAYER_COLOR
                center = (x * CELL_SIZE + CELL_SIZE // 2,
                         y * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(screen, color, center, STONE_RADIUS)

def print_startup_info():
    print("\nPyGomoku v1.0")
    print("==================")
    print("Controls:")
    print("- Mouse click: Place your stone")
    print("- Backspace:   Take back 2 moves (your last move and bot's response)")
    print("- Close window to quit")
    print("\nNotation:")
    print("- Columns: A-Y (left to right)")
    print("- Rows: 1-25 (bottom to top)")
    print("\nGame starting...\n")


def main():
    # Print startup information
    print_startup_info()

    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    game_over = False
    last_bot_move = None
    move_number = 1
    
    # Store move history for takebacks
    move_history = []  # Will store tuples of (x, y, player_value)
    
    # Bot's first move if it plays first
    if BOT_PLAYS_FIRST:
        x, y = improved_bot_move(board, BOT_PLAYS_FIRST, MAX_PLY_DEPTH)
        board[y][x] = 1
        last_bot_move = (x, y)
        print(f"{move_number}. bot: {coord_to_notation(x, y)}")
        move_history.append((x, y, 1))
        move_number += 1
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                # Take back two moves if possible (human's last move and bot's response)
                if len(move_history) >= 2:
                    # Remove last two moves
                    for _ in range(2):
                        x, y, _ = move_history.pop()
                        board[y][x] = 0
                    
                    # Update last bot move
                    if move_history:
                        last_bot_move = (move_history[-1][0], move_history[-1][1])
                    else:
                        last_bot_move = None
                    
                    # Update move number and game state
                    move_number = max(1, move_number - 1)
                    game_over = False
                    print("Took back 2 moves")
                    print(f"Position after {len(move_history)} moves")
                
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                board_x = mouse_x // CELL_SIZE
                board_y = mouse_y // CELL_SIZE
                
                if board[board_y][board_x] == 0:
                    # Player's move
                    player_value = -1 if BOT_PLAYS_FIRST else 1
                    board[board_y][board_x] = player_value
                    print(f"{move_number}. human: {coord_to_notation(board_x, board_y)}")
                    move_history.append((board_x, board_y, player_value))
                    
                    draw_board(last_bot_move)
                    draw_stones(board)
                    pygame.display.flip()
                    
                    if check_winner(board, board_x, board_y, player_value):
                        print("Player wins!")
                        game_over = True
                    else:
                        print("Bot is thinking...")
                        start_time = time.time()
                        x, y = improved_bot_move(board, BOT_PLAYS_FIRST, MAX_PLY_DEPTH)
                        end_time = time.time()
                        print(f"Bot moved in {end_time - start_time:.2f} seconds")
                        print(f"{move_number}. bot: {coord_to_notation(x, y)}")
                        
                        # Bot's move
                        bot_value = 1 if BOT_PLAYS_FIRST else -1
                        board[y][x] = bot_value
                        last_bot_move = (x, y)
                        move_history.append((x, y, bot_value))
                        
                        if check_winner(board, x, y, bot_value):
                            print("Bot wins!")
                            game_over = True
                        
                        move_number += 1
        
        # Regular board updates
        draw_board(last_bot_move)
        draw_stones(board)
        pygame.display.flip()

if __name__ == "__main__":
    main()


