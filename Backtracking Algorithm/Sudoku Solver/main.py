import time
import pyautogui as pg


class SolveSudoku:

    def find_empty(self, puzzle):

        for row in range(9):
            for col in range(9):
                if puzzle[row][col] == 0:
                    return row, col

        return None, None  # if no spaces

    def validity(self, puzzle, guess, row, col):

        row_val = puzzle[row]  # check in row
        if guess in row_val:  # if the guess is already exist in the row ignore, and loop again for another value
            return False

        col_val = [puzzle[i][col] for i in range(9)]  # check in col
        if guess in col_val:   # if the guess is already exist in the column ignore, and loop again for another value
            return False

        row_start = (row // 3) * 3  # check in box 3*3
        col_start = (col // 3) * 3
        for r in range(row_start, row_start + 3):
            for c in range(col_start, col_start + 3):
                if puzzle[r][c] == guess:   # puzzle[3][0]
                    return False
        return True

    def autoSolver(self, matrix):
        final = []
        str_fin = []
        for i in range(9):
            final.append(matrix[i])

        for lists in final:
            for num in lists:
                str_fin.append(str(num))

        counter = []

        for num in str_fin:
            pg.press(num)
            pg.hotkey('right')
            counter.append(num)
            if len(counter) % 9 == 0:
                pg.hotkey('down')
                for i in range(0, 8):
                    pg.hotkey('left')

    def solve_sudoku(self):

        row, col = self.find_empty(grid)
        if row is None:
            return True

        for guess in range(1, 10):  # loop from 1 to 9 (Available Values)
            if self.validity(grid, guess, row, col):  # validity => check if the value is valid in each empty cell
                grid[row][col] = guess
                if self.solve_sudoku():
                    return True
            grid[row][col] = 0  # if the value is not valid so we need to backtrack and try new value
        return False   # if we didn't find the solution


if __name__ == '__main__':
    obj = SolveSudoku()
    grid = []

    while True:
        row = list(input('Row: '))
        ints = []

        for n in row:
            ints.append(int(n))
        grid.append(ints)

        if len(grid) == 9:
            break
        print('Row ' + str(len(grid)) + ' Complete')
    time.sleep(3)
    start = time.time()
    if obj.solve_sudoku():
        end = time.time()
        obj.autoSolver(grid)
    print(grid)
print(f"Solved In {round(end - start, 2)}S!")
