from copy import deepcopy
import random


def valid_matrix(matrix):
    if not isinstance(matrix, (list, tuple)):
        raise Exception('it not matrix')
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            raise Exception('it not matrix')
        elif len(row) != len(matrix[0]):
            raise Exception('rows have not same length')
        for elem in row:
            if not isinstance(elem, (int, float)):
                raise Exception('matrix element not number')


class Matrix:
    def __init__(self, matrix):
        valid_matrix(matrix)
        self.data = deepcopy(matrix)
        self._height = len(self.data)
        self._width = len(self.data[0])

    def __add__(self, other):
        if not self.same_dimension_with(other):
            raise ValueError('Matrices have different sizes, adding is impossible')
        new_data = [[self.data[i][j] + other.data[i][j] for j in range(self._width)] for i in range(self._height)]
        return Matrix(new_data)

    def __sub__(self, other):
        if not self.same_dimension_with(other):
            raise ValueError('Matrices have different sizes, adding is impossible')
        new_data = [[self.data[i][j] - other.data[i][j] for j in range(self._width)] for i in range(self._height)]
        return Matrix(new_data)

    def __mul__(self, other):
        if not isinstance(other, (Matrix, int, float)):
            raise TypeError(f'Matrix cannot be multiplied by {type(other)}')
        if isinstance(other, Matrix):
            if self._width != other._height or self._height != other._width:
                raise ValueError('The number of columns in first matrix must match the number of rows in second matrix')
            new_data = [[sum(i * j for i, j in zip(i_row, j_col)) for j_col in zip(*other.data)] for i_row in self.data]
        else:
            new_data = [[other * self.data[i][j] for j in range(self._width)] for i in range(self._height)]
        return Matrix(new_data)

    def __str__(self):
        res = []
        for index, row in enumerate(self.data):
            if index == 0:
                before, after = '⌈', '⌉'
            elif index == self._height - 1:
                before, after = '⌊', '⌋'
            else:
                before = after = '|'
            standard = max(map(len, (str(x) for x in sum(self.data, []))))
            res.append(f"{before} {' '.join(str(elem).center(standard) for elem in row)} {after}")
        result = '\n'.join(res)
        return result

    def determinant(self):
        if not self.is_square():
            raise ValueError('Matrix must be square')
        my_matrix = list(map(list, deepcopy(self.data)))
        for index in range(self._width):
            for i in range(index + 1, self._width):
                if my_matrix[index][index] == 0:
                    my_matrix[index][index] = 1.0e-14
                scalar = my_matrix[i][index] / my_matrix[index][index]
                for j in range(self._height):
                    my_matrix[i][j] = my_matrix[i][j] - scalar * my_matrix[index][j]
        result = 1
        for index in range(self._width):
            result *= my_matrix[index][index]
        return round(result, 10)

    def inverse(self):
        if self.determinant() == 0:
            raise ValueError('matrix not invertible, determinant = 0')
        compliment_matrix = [[0 for _ in range(self._width)] for _ in range(self._height)]
        for i in range(self._height):
            tmp_matrix = deepcopy(self.data)
            tmp_matrix.pop(i)
            tmp_matrix = list(zip(*tmp_matrix))
            for j in range(self._width):
                copy_tmp_matrix = deepcopy(tmp_matrix)
                copy_tmp_matrix.pop(j)
                minor = list(zip(*copy_tmp_matrix))
                compliment_matrix[i][j] = Matrix(minor).determinant() * (-1)**(i+j)
        transposed_compliment_matrix = list(zip(*compliment_matrix))
        inverse_matrix = Matrix(transposed_compliment_matrix) * (1 / self.determinant())
        return inverse_matrix

    def same_dimension_with(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f'"{other}" must be Matrix')
        if self._width == other._width and self._height == other._height:
            return True
        return False

    def is_square(self):
        if self._width == self._height:
            return True
        return False

    @staticmethod
    def random_matrix(count_of_rows, count_of_columns):
        mu = random.randint(1, 100)
        sigma = random.randint(1, 100000)
        random_data = [[random.normalvariate(mu, sigma) for _ in range(count_of_columns)] for _ in range(count_of_rows)]
        return Matrix(random_data)
