import sys
import timeit
import itertools

global nodes_expanded



# I/O Helpers:

def read_puzzle_file(filename):
    puzzle = []
    with open(filename) as pf:
        for i, line in enumerate(pf):
            puzzle.append([])
            for _, el in enumerate(line.split(' ')):
                puzzle[i].append(int(el))
    return puzzle


def write_puzzle(puzzle, filename):
    performance = open(filename, "w+")
    row = 0
    while row < 9:
        column = 0
        while column < 9:
            if column < 8:
                performance.write(str(puzzle[row][column]) + " ")
            else:
                performance.write(str(puzzle[row][column]) + "\n")
            column += 1
        row += 1
    performance.close()
    return True


def write_performance(algorithm, puzzlefile, totaltime, searchtime, nodesnum, filename):
    pf = open(filename, "w+")
    pf.write("Performance of the algorithm: " + algorithm + " \n")
    pf.write("[ Performance ] total time      : " + str(totaltime*1000) + " ms\n")
    pf.write("                search time     : " + str(searchtime*1000) + " ms\n")
    pf.write("                nodes generated : " + str(nodesnum) + " \n")
    pf.close()
    return True


def print_puzzle(puzzle):
    output = "\t\t"
    row = 0
    while row < 9:
        column = 0
        while column < 9:
            if column < 8:
                output += str(puzzle[row][column]) + " "
            else:
                output += str(puzzle[row][column]) + "\n\t\t"
            column += 1
        row += 1
    print(output)


def print_performance(algorithm, puzzlefile, puzzle, totaltime, searchtime, nodesnum):
    print("Solution for " + puzzlefile + " using " + algorithm + ": ")
    print("[Result Matrix]")
    print_puzzle(puzzle)
    print("[ Performance ] total time      : " + str(totaltime*1000) + " ms")
    print("                search time     : " + str(searchtime*1000) + " ms")
    print("                nodes generated : " + str(nodesnum) + " \n")
    return True



# algorithm helpers:

def judge(puzzle):
    # judge row
    for row in puzzle:
        row_set = set(row)
        if 0 in row_set or len(row_set) != 9:
            return False
    # judge column
    i = 0
    while i < 9:
        column_set = set([puzzle[0][i], puzzle[1][i], puzzle[2][i], 
                          puzzle[3][i], puzzle[4][i], puzzle[5][i], 
                          puzzle[6][i], puzzle[7][i], puzzle[8][i]])
        if 0 in column_set or len(column_set) != 9:
            return False
        i += 1
    # judge block
    block_centers = [(1,1), (1,4), (1,7), (4,1), (4,4), (4,7), (7,1), (7,4), (7,7)]
    i = 0
    while i < 9:
        r = block_centers[i][0]
        c = block_centers[i][1]
        block_set = set([puzzle[r-1][c-1], puzzle[r-1][c], puzzle[r-1][c+1], 
                         puzzle[r][c-1], puzzle[r][c], puzzle[r][c+1], 
                         puzzle[r+1][c-1], puzzle[r+1][c], puzzle[r+1][c+1]])
        if 0 in block_set or len(block_set) != 9:
            return False
        i += 1
    return True


def get_groups(puzzle, cell):
    r = cell[0]
    c = cell[1]

    row = puzzle[r]

    column = []
    for i in range(0, 9):
        column.append(puzzle[i][c])

    block = []
    center_r = int(r/3) * 3 + 1 
    center_c = int(c/3) * 3 + 1
    for i in range(-1, 2):
        for j in range(-1, 2):
            block.append(puzzle[center_r+i][center_c+j])

    return [row, column, block]


def get_incomplete_cells(puzzle):
    cells = []
    row = 0
    while row < 9:
        column = 0
        while column < 9:
            if puzzle[row][column] == 0:
                cells.append((row, column))
            column += 1
        row += 1
    return cells


def get_possible_numers(puzzle, cell):
    possibles = []
    groups = get_groups(puzzle, cell)
    for num in range(1, 10):
        if num not in groups[0] and num not in groups[1] and num not in groups[2]:
            possibles.append(num)
    return possibles


def get_mrv(puzzle):
    mrv_cells = []
    row = 0
    while row < 9:
        column = 0
        while column < 9:
            if puzzle[row][column] == 0:
                cell = (row, column)
                possibles = get_possible_numers(puzzle, cell)
                mrv_cells.append((len(possibles), cell, possibles))
            column += 1
        row += 1
    return sorted(mrv_cells)



# algorithms

def prune_problem(puzzle):
    incomplete_cells = get_incomplete_cells(puzzle) 
    prune_time = 9
    while prune_time > 0:
        for cell in incomplete_cells:
            possibles = get_possible_numers(puzzle, cell)
            if len(possibles) == 1:
                print("pruned one branch!")
                puzzle[cell[0]][cell[1]] = possibles[0]
                incomplete_cells = get_incomplete_cells(puzzle)
        prune_time -= 1

def brute_force(puzzle):
    incomplete_cells = get_incomplete_cells(puzzle)         
    possible_set = []
    for cell in incomplete_cells:
        possibles = get_possible_numers(puzzle, cell)
        possible_set.append(possibles)

    search_set = itertools.product(*possible_set)

    for search_iter in search_set:
        for i, cell in enumerate(incomplete_cells):
            puzzle[cell[0]][cell[1]] = search_iter[i]
            global nodes_expanded
            nodes_expanded += 1
        
        # print_puzzle(puzzle)
        if judge(puzzle):
            write_puzzle(puzzle, solution_file)
            return True
    
    return False


def back_tracking(puzzle, cells):
    if judge(puzzle):
        write_puzzle(puzzle, solution_file)
        return True
    if len(cells) == 0:
        return False
    
    c = cells[0]
    for num in get_possible_numers(puzzle, c):
        global nodes_expanded
        nodes_expanded += 1

        puzzle[c[0]][c[1]] = num
        # print_puzzle(puzzle)
        if back_tracking(puzzle, cells[1:]):
            return True
        puzzle[c[0]][c[1]] = 0
    
    return False


def forward_checking_mrv(puzzle):
    if judge(puzzle):
        write_puzzle(puzzle, solution_file)
        return True

    incomplete_cells = get_mrv(puzzle)
    if len(incomplete_cells) == 0:
        return False
    cell = incomplete_cells[0]
    c = cell[1]
    possibles = cell[2]

    for num in possibles:
        global nodes_expanded
        nodes_expanded += 1

        puzzle[c[0]][c[1]] = num
        if forward_checking_mrv(puzzle):
            return True
        puzzle[c[0]][c[1]] = 0

    return False



# Main

if __name__ == "__main__":
    total_time_start = timeit.default_timer()
    nodes_expanded = 0

    if len(sys.argv) != 3:
        print ("[ERR] Usage: Python SudokuSolver.py <puzzle_file> <algorithm: BF/BT/FC-MRV>")
        print ("[ERR] Please check the command and run again, make sure 2 arguments assigned.")
    else :
        puzzle_file = sys.argv[1]
        algorithm = sys.argv[2]
        print ("Solving " + puzzle_file + " using " + algorithm + "...")
        solution_file = "./solutions/solution" + puzzle_file[-5:]
        performance_file = "./performances/performance" + puzzle_file[-5:]
        puzzle = read_puzzle_file(puzzle_file)
        print_puzzle(puzzle)

        if algorithm == "BF":
            search_time_start = timeit.default_timer()
            # prune_problem(puzzle)
            flag = brute_force(puzzle)
            search_time_stop = timeit.default_timer()

        elif algorithm == "BT":
            search_time_start = timeit.default_timer()
            # prune_problem(puzzle)
            incomplete_cells = get_incomplete_cells(puzzle)
            flag = back_tracking(puzzle, incomplete_cells)
            search_time_stop = timeit.default_timer()

        elif algorithm == "FC-MRV":
            search_time_start = timeit.default_timer()
            # prune_problem(puzzle)
            flag = forward_checking_mrv(puzzle)
            search_time_stop = timeit.default_timer()

        else:
            print("[ERR]Optional algorithms are: BF, BT, FC-MRV.")

    total_time_stop = timeit.default_timer()
    total_time = total_time_stop - total_time_start
    search_time = search_time_stop - search_time_start

    if flag:
        print("algorithm done computation successfully!")
        print("Solution and performance are saved to relative folder.")
        print("You can also check it below:")
    print_performance(algorithm, puzzle_file, puzzle, total_time, search_time, nodes_expanded)
    write_performance(algorithm, puzzle_file, total_time, search_time, nodes_expanded, performance_file)