#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>
#include <random>
#include <chrono>

class Matrix {
public:
	// Construstores
	Matrix(int rows, int cols);
	Matrix(std::initializer_list<std::initializer_list<double>> values);

	// Getters & Setters
	double getValue(int row, int col) const;
	void setValue(int row, int col, double value);

	int getRows() const;
	int getCols() const;

	// Operações aritméticas
	Matrix operator*(const double scalar) const;
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;

	// Operações matriciais
	static Matrix dot(const Matrix& matrixA, const Matrix& matrixB);
	Matrix hadamard(const Matrix& other) const;
	Matrix T() const;

	// Utilitários
	double Matrix::sum() const;
	void print() const;
	void randomize();
	void map(std::function<double(int, int, double)> modifier);
	Matrix normalize() const;

private:
	int rows;
	int cols;
	std::vector<std::vector<double>> data;
};

#endif