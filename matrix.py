from copy import deepcopy
from random import randint, normalvariate
from typing import Union


def valid_matrix(matrix: Union[list, tuple]) -> None:
    """
    That function raise exception if param matrix not 2d_array,
                                  or elements in 2d_array not numbers,
                                  or length of rows not same
    :param matrix: 2d_array
    :return: None
    """
    if not isinstance(matrix, (list, tuple)):
        raise TypeError('it not matrix')
    for row in matrix:
        if not isinstance(row, (list, tuple)):
            raise TypeError('it not matrix')
        elif len(row) != len(matrix[0]):
            raise ValueError('rows have not same length')
        for elem in row:
            if not isinstance(elem, (int, float)):
                raise ValueError('matrix element not number')


class Matrix:
    def __init__(self, matrix: Union[list, tuple]):
        """
        Initialize Matrix object
        :param matrix: 2d_array with elements data
        :type matrix: tuple or list
        :raises TypeError: if matrix not list or tuple of lists or tuples
        :raises ValueError: if elements not numbers, or length of rows not same
        """
        valid_matrix(matrix)
        self.data = deepcopy(matrix)
        self._height = len(self.data)
        self._width = len(self.data[0])

    def __add__(self, other):
        """
        Define addition for Matrix objects.
        :param other: matrix which addition to original matrix.
        :type other: Matrix
        :return: new matrix = original matrix + other matrix
        :rtype: Matrix
        :raises ValueError: if matrices have different dimension
        :raises TypeError: if type of "other" not Matrix
        """
        if not self.same_dimension_with(other):
            raise ValueError('Matrices have different sizes, adding is impossible')
        new_data = [[self.data[i][j] + other.data[i][j] for j in range(self._width)] for i in range(self._height)]
        return Matrix(new_data)

    def __sub__(self, other):
        """
        Define subtraction for Matrix objects.
        :param other: matrix which subtract from original matrix.
        :type other: Matrix
        :return: new matrix = original matrix - other matrix
        :rtype: Matrix
        :raise ValueError: if matrices have different dimension
        :raises TypeError: if type of "other" not Matrix
        """
        if not self.same_dimension_with(other):
            raise ValueError('Matrices have different sizes, adding is impossible')
        new_data = [[self.data[i][j] - other.data[i][j] for j in range(self._width)] for i in range(self._height)]
        return Matrix(new_data)

    def __mul__(self, other):
        """
        Define multiplication for Matrix objects.
        :param other: matrix, or number for which you multiplying original matrix.
        :type  other: Matrix, int, float
        :return: in case other type is matrix : new matrix = original matrix * other matrix,
                 in case other type is number : new matrix = original matrix * number
        :rtype: Matrix
        :raises ValueError: if number of columns in first matrix don't match the number of rows in second matrix
        :raises TypeError: if type of "other" not Matrix or int or float
        """
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
        """
        Method for visualization matrix, and make Matrix object printable.
        :return:string which visualize matrix
        :rtype: str
        """
        res = []
        for index, row in enumerate(self.data):
            if index == 0:
                before, after = '⌈', '⌉'
            elif index == self._height - 1:
                before, after = '⌊', '⌋'
            else:
                before = after = '|'
            standard = max(map(len, (str(x) for x in sum(self.data, []))))  # take as standard the longest element
            res.append(f"{before} {' '.join(str(elem).center(standard) for elem in row)} {after}")
        result = '\n'.join(res)
        return result

    def determinant(self):
        """
        Method which calculates determinant of matrix by using method of reducing to triangular form.
        :return: determinant of the matrix
        :rtype: int, float
        :raise ValueError: if matrix not square
        """
        if not self.is_square():
            raise ValueError('Matrix must be square')
        tmp_matrix = list(map(list, deepcopy(self.data)))  # change tuples in data to list, for make them mutable
        # bring the tmp_matrix into a triangular form
        for index in range(self._width):
            for i in range(index + 1, self._width):
                if tmp_matrix[index][index] == 0:  # if diagonal element equal zero, change it to approximately zero
                    tmp_matrix[index][index] = 1.0e-14  # in other case will be ZeroDivisionError
                scalar = tmp_matrix[i][index] / tmp_matrix[index][index]
                for j in range(self._height):
                    tmp_matrix[i][j] = tmp_matrix[i][j] - scalar * tmp_matrix[index][j]
        # the determinant equal to the product of diagonal elements of the triangular matrix
        result = 1
        for index in range(self._width):
            result *= tmp_matrix[index][index]
        # need to approximate the result to avoid the tails of the product of floating point numbers
        return round(result, 10)

    def inverse(self):
        """
        Method which find inverse matrix by using matrix of algebraic complements.
        :return: the inverse of the original matrix
        :rtype: Matrix
        :raise ValueError: if matrix not square, or determinant equal to 0
        """
        if self.determinant() == 0:
            raise ValueError('Matrix not invertible, determinant = 0')
        # create matrix with same dimension and fill it with 0
        compliment_matrix = [[0 for _ in range(self._width)] for _ in range(self._height)]
        for i in range(self._height):
            tmp_matrix = deepcopy(self.data)
            tmp_matrix.pop(i)  # remove from tmp_matrix row with number i
            tmp_matrix = list(zip(*tmp_matrix))  # transpose tmp_matrix
            for j in range(self._width):
                copy_tmp_matrix = deepcopy(tmp_matrix)
                copy_tmp_matrix.pop(j)  # remove row j in transposed matrix equal to remove column j in original matrix
                minor = list(zip(*copy_tmp_matrix))  # transpose back to find minor[i][j] of the original matrix
                compliment_matrix[i][j] = Matrix(minor).determinant() * (-1)**(i+j)
        transposed_compliment_matrix = list(zip(*compliment_matrix))
        inverse_matrix = Matrix(transposed_compliment_matrix) * (1 / self.determinant())
        return inverse_matrix

    def same_dimension_with(self, other):
        """
        Method which checks matrices have same dimension or not
        :param other: the matrix which we compare with original matrix
        :type other: Matrix
        :return: True if they have same dimension, False if have not
        :rtype: bool
        :raise TypeError: if type of "other" not Matrix
        """
        if not isinstance(other, Matrix):
            raise TypeError(f'"{other}" must be Matrix')
        if self._width == other._width and self._height == other._height:
            return True
        return False

    def is_square(self):
        """
        Method which checks matrix is square or not
        :return: True if it square, False if it not
        :rtype: bool
        """
        if self._width == self._height:
            return True
        return False

    @staticmethod
    def random_matrix(count_of_rows, count_of_columns, mu=None, sigma=None):
        """
        Method which create a random matrix whose entries are random numbers, subject to normal distribution.
        :param count_of_rows: how many rows must be in random matrix
        :type count_of_rows: int
        :param count_of_columns:how many columns must be in random matrix
        :type count_of_columns: int
        :param mu: mean . Optional parameter. If not given take random from 1 to 100
        :type mu: int or float
        :param sigma: standard deviation. Optional parameter. if not given take random from 1 to 100000
        :type sigma: int or float
        :return: matrix with random elements
        :rtype: Matrix
        :raise TypeError: if any parameter type does not match
        """
        if not isinstance(count_of_rows, int) or not isinstance(count_of_columns, int):
            raise TypeError('count of rows and columns must be integer')
        if not isinstance(mu, (int, float, type(None))) or not isinstance(sigma, (int, float, type(None))):
            raise TypeError('mu and sigma must be integer or float')
        if mu is None:
            mu = randint(1, 100)
        if sigma is None:
            sigma = randint(1, 100000)
        random_data = [[normalvariate(mu, sigma) for _ in range(count_of_columns)] for _ in range(count_of_rows)]
        return Matrix(random_data)
