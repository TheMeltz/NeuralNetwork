#include "Matrix.h"
#include <stdexcept>

// Construtores
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values) {
    rows = values.size();
    if (rows > 0) {
        cols = values.begin()->size();
        data.resize(rows, std::vector<double>(cols, 0.0));

        int i = 0;
        for (const auto& row : values) {
            if (row.size() != static_cast<size_t>(cols)) {
                throw std::invalid_argument("Inconsistent column size in initializer list");
            }
            int j = 0;
            for (double value : row) {
                data[i][j] = value;
                j++;
            }
            i++;
        }
    } else {
        cols = 0;
    }
}

// Operações aritméticas
Matrix Matrix::operator*(const double scalar) const {
	Matrix result(getRows(), getCols());

	for (int i = 0; i < getRows(); i++) {
		for (int j = 0; j < getCols(); j++) {
			result.setValue(i, j, getValue(i, j) * scalar);
		}
	}

	return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.getRows() || cols != other.getCols()) {
        throw std::invalid_argument("Incompatible matrix dimensions for addition");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.setValue(i, j, data[i][j] + other.getValue(i, j));
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.getRows() || cols != other.getCols()) {
        throw std::invalid_argument("Incompatible matrix dimensions for subtraction");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.setValue(i, j, data[i][j] - other.getValue(i, j));
        }
    }

    return result;
}

// Operações matriciais
Matrix Matrix::dot(const Matrix& matrixA, const Matrix& matrixB) {
	if (matrixA.getCols() != matrixB.getRows()) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    Matrix result(matrixA.getRows(), matrixB.getCols());

    for (int i = 0; i < matrixA.getRows(); i++) {
        for (int j = 0; j < matrixB.getCols(); j++) {
            double sum = 0.0;
            for (int k = 0; k < matrixA.getCols(); k++) {
                sum += matrixA.getValue(i, k) * matrixB.getValue(k, j);
            }
            result.setValue(i, j, sum);
        }
    }

    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (getRows() != other.getRows() || getCols() != other.getCols()) {
        throw std::invalid_argument("Incompatible matrix dimensions for Hadamard product");
    }

    Matrix result(getRows(), getCols());

    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            result.setValue(i, j, getValue(i, j) * other.getValue(i, j));
        }
    }

    return result;
}

Matrix Matrix::T() const {
	Matrix result(cols, rows);

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			result.setValue(col, row, data[row][col]);
		}
	}

	return result;
}

// Utilitários
double Matrix::sum() const {
    double total = 0.0;
    for (const auto& row : data) {
        for (double value : row) {
            total += value;
        }
    }
    return total;
}

void Matrix::print() const {
    std::cout << "{\n";
    for (const auto& row : data) {
        std::cout << "\t{";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << row[i];
            if (i < row.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}\n";
    }
    std::cout << "}\n";
}

void Matrix::randomize() {
    std::random_device rd;
    std::mt19937::result_type seed = rd() ^ (
        (std::mt19937::result_type)
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count() +
        (std::mt19937::result_type)
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count());
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    map([&](int row, int col, double element) {
        return distribution(gen);
    });
}

void Matrix::map(std::function<double(int, int, double)> modifier) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = modifier(i, j, data[i][j]);
        }
    }
}

Matrix Matrix::normalize() const {
    Matrix result(getRows(), getCols());
    for (int col = 0; col < getCols(); ++col) {
        double maxVal = getValue(0, col);

        for (int row = 1; row < getRows(); ++row) {
            if (getValue(row, col) > maxVal) {
                maxVal = getValue(row, col);
            }
        }

        for (int row = 0; row < getRows(); ++row) {
            result.setValue(row, col, getValue(row, col) / maxVal);
        }
    }
    return result;
}

// Getters & Setters
double Matrix::getValue(int row, int col) const {
	return data[row][col];
}

void Matrix::setValue(int row, int col, double value) {
	data[row][col] = value;
}

int Matrix::getRows() const {
	return rows;
}

int Matrix::getCols() const {
	return cols;
}