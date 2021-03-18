import random
import operator
from past.builtins import range

import numpy as np

NumOfDigits = 9


class Population(object):
    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, original):
        helper = Candidate()
        helper.values = [[[] for i in range(0, NumOfDigits)] for j in range(0, NumOfDigits)]
        for row in range(0, NumOfDigits):
            for col in range(0, NumOfDigits):
                for value in range(1, 10):  # Available Values
                    if (original.values[row][col] == 0) and \
                            not (original.is_column_duplicate(col, value)
                                 or original.is_row_duplicate(row, value)
                                 or original.is_block_duplicate(row, col, value)):
                        helper.values[row][col].append(value)
                    elif original.values[row][col] != 0:
                        helper.values[row][col].append(original.values[row][col])
                        break
        # helper is a list that contains all the possible values in each cell
        # print(helper.values)

        for p in range(0, Nc):
            g = Candidate()
            for i in range(0, NumOfDigits):
                row = np.zeros(NumOfDigits)
                for j in range(0, NumOfDigits):
                    if original.values[i][j] != 0:
                        row[j] = original.values[i][j]
                    elif original.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]
                # print(row)

                q = 0
                while len(list(set(row))) != NumOfDigits:
                    q += 1
                    if q > 500000:
                        return 0
                    for j in range(0, NumOfDigits):
                        if original.values[i][j] == 0:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]
                g.values[i] = row
                # print(row)
            self.candidates.append(g)

        # self.candidates = [list(x) for x in set(tuple(x) for x in self.candidates)]
        self.update_fitness()

        return 1

    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return

    def sort(self):
        """ Sort the population based on fitness. """
        self.candidates = sorted(self.candidates, key=operator.attrgetter('fitness'))
        return


class Candidate(object):
    def __init__(self):
        self.values = np.zeros((NumOfDigits, NumOfDigits))
        self.fitness = None
        return

    def update_fitness(self):
        col_count, block_count = np.zeros(NumOfDigits), np.zeros(NumOfDigits)
        col_sum, block_sum = 0, 0
        self.values = self.values.astype(int)

        # for each column
        for i in range(0, NumOfDigits):
            for j in range(0, NumOfDigits):
                col_count[self.values[i][j] - 1] += 1
                #print(col_count)
            for k in range(len(col_count)):
                if col_count[k] == 1:
                    col_sum += (1 / NumOfDigits) / NumOfDigits
            col_count = np.zeros(NumOfDigits)

        # for each block
        for i in range(0, NumOfDigits, 3):
            for j in range(0, NumOfDigits, 3):
                block_count[self.values[i][j] - 1] += 1
                block_count[self.values[i][j + 1] - 1] += 1
                block_count[self.values[i][j + 2] - 1] += 1

                block_count[self.values[i + 1][j] - 1] += 1
                block_count[self.values[i + 1][j + 1] - 1] += 1
                block_count[self.values[i + 1][j + 2] - 1] += 1

                block_count[self.values[i + 2][j] - 1] += 1
                block_count[self.values[i + 2][j + 1] - 1] += 1
                block_count[self.values[i + 2][j + 2] - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1 / NumOfDigits) / NumOfDigits
                block_count = np.zeros(NumOfDigits)

        if int(col_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = col_sum * block_sum
        self.fitness = fitness
        return

    def mutate(self, mutation_rate, given):
        """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

        r = random.uniform(0, 1.1)
        while r > 1:  # Outside [0, 1] boundary - choose another
            r = random.uniform(0, 1.1)

        success = False
        if r < mutation_rate:  # Mutate.
            while not success:
                row1 = random.randint(0, 8)
                row2 = random.randint(0, 8)
                row2 = row1

                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while from_column == to_column:
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                    # Check if the two places are free to swap
                if given.values[row1][from_column] == 0 and given.values[row1][to_column] == 0:
                    # ...and that we are not causing a duplicate in the rows' columns.
                    if not given.is_column_duplicate(to_column, self.values[row1][from_column]) \
                            and not given.is_column_duplicate(from_column, self.values[row2][to_column]) \
                            and not given.is_block_duplicate(row2, to_column, self.values[row1][from_column]) \
                            and not given.is_block_duplicate(row1, from_column, self.values[row2][to_column]):
                        # Swap values.
                        temp = self.values[row2][to_column]
                        self.values[row2][to_column] = self.values[row1][from_column]
                        self.values[row1][from_column] = temp
                        success = True

        return success


class Rules(Candidate):
    # a method that passing given values to check

    def __init__(self, values):
        self.values = values
        return

    # a methods that check if digits is duplicated in a same row or column
    # value = my value which i want to add
    # return (true) if digits is duplicated

    def is_row_duplicate(self, row, value):

        for column in range(0, NumOfDigits):
            if self.values[row][column] == value:
                return True
        return False

    def is_column_duplicate(self, column, value):
        for row in range(0, NumOfDigits):
            if self.values[row][column] == value:
                return True
        return False

    # check duplicate between digits in a same small 3*3 block

    def is_block_duplicate(self, row, column, value):

        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))
        if ((self.values[i][j] == value)
                or (self.values[i][j + 1] == value)
                or (self.values[i][j + 2] == value)
                or (self.values[i + 1][j] == value)
                or (self.values[i + 1][j + 1] == value)
                or (self.values[i + 1][j + 2] == value)
                or (self.values[i + 2][j] == value)
                or (self.values[i + 2][j + 1] == value)
                or (self.values[i + 2][j + 2] == value)):

            return True
        else:
            return False


class Tournament(object):
    def __init__(self):
        return

    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if f1 > f2:
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        # selection_rate = 0.85
        selection_rate = 0.80
        r = random.uniform(0, 1.1)
        while r > 1:  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
        if r < selection_rate:
            return fittest
        else:
            return weakest


class CycleCrossover(object):
    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate()
        child2 = Candidate()

        # Make a copy of the parent genes.
        child1.values = np.copy(parent1.values)
        child2.values = np.copy(parent2.values)

        r = random.uniform(0, 1.1)
        while r > 1:  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)

        # Perform crossover.
        if r < crossover_rate:
            # Pick a crossover point. Crossover must have at least 1 row (and at most Nd-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while crossover_point1 == crossover_point2:
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)

            if crossover_point1 > crossover_point2:
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2):
        child_row1 = np.zeros(NumOfDigits)
        child_row2 = np.zeros(NumOfDigits)

        remaining = range(1, NumOfDigits + 1)
        cycle = 0

        while (0 in child_row1) and (0 in child_row2):  # While child rows not complete...
            if cycle % 2 == 0:  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]

                while next != start:  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]

                while next != start:  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]

                cycle += 1

        return child_row1, child_row2

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if parent_row[i] in remaining:
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if parent_row[i] == value:
                return i


class Sudoku(object):
    def __init__(self):
        self.original = None
        return

    def load(self, p):
        self.original = Rules(p)
        return

    def solve(self):
        Nc = 1000  # Number of candidates (i.e. population size).
        Ne = int(0.05 * Nc)  # Number of elites. // 50
        Ng = 10000  # Number of generations.
        Nm = 0  # Number of mutations.

        # Mutation parameters.
        phi = 0
        sigma = 1
        mutation_rate = 0.06

        self.population = Population()
        print("Create An Initial Population")
        if self.population.seed(Nc, self.original):
            pass

        # print(self.population.candidates[0])

        stale = 0
        for generation in range(0, Ng):

            # Check for a solution.
            best_fitness = 0.0
            # best_fitness_population_values = self.population.candidates[0].values
            for c in range(0, Nc):
                fitness = self.population.candidates[c].fitness
                if (fitness == 1):
                    print("Solution found at generation %d!" % generation)
                    return (generation, self.population.candidates[c])

                # Find the best fitness and corresponding chromosome
                if (fitness > best_fitness):
                    best_fitness = fitness
                    # best_fitness_population_values = self.population.candidates[c].values

            print("Generation:", generation, " Best fitness:", best_fitness)
            # print(best_fitness_population_values)

            # Create the next population.
            next_population = []

            # Select elites (the fittest candidates) and preserve them for the next generation.
            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = np.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Create the rest of the candidates.
            for count in range(Ne, Nc, 2):
                # Select parents from population via a tournament.
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                # Cross-over.
                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                # Mutate child1.
                child1.update_fitness()
                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.original)
                child1.update_fitness()
                if success:
                    Nm += 1
                    if child1.fitness > old_fitness:  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1

                # Mutate child2.
                child2.update_fitness()
                old_fitness = child2.fitness
                success = child2.mutate(mutation_rate, self.original)
                child2.update_fitness()
                if (success):
                    Nm += 1
                    if (child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1

                # Add children to new population.
                next_population.append(child1)
                next_population.append(child2)

            # Append elites onto the end of the population. These will not have been affected by crossover or mutation.
            for e in range(0, Ne):
                next_population.append(elites[e])

            # Select next generation.
            self.population.candidates = next_population
            self.population.update_fitness()

            # Calculate new adaptive mutation rate (based on Rechenberg's 1/5 success rule).
            # This is to stop too much mutation as the fitness progresses towards unity.
            if Nm == 0:
                phi = 0  # Avoid divide by zero.
            else:
                phi = phi / Nm

            if phi > 0.2:
                sigma = sigma / 0.998
            elif phi < 0.2:
                sigma = sigma * 0.998

            mutation_rate = abs(np.random.normal(loc=0.0, scale=sigma, size=None))

            # Check for stale population.
            self.population.sort()
            if self.population.candidates[0].fitness != self.population.candidates[1].fitness:
                stale = 0
            else:
                stale += 1

            # Re-seed the population if 100 generations have passed
            # with the fittest two candidates always having the same fitness.
            if stale >= 100:
                print("The population has gone stale. Re-seeding...")
                self.population.seed(Nc, self.original)
                stale = 0
                sigma = 1
                phi = 0
                mutation_rate = 0.06

        print("No solution found.")
        return -2, 1
