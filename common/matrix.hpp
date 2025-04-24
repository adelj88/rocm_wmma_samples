/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef HIP_MATRIX_HPP
#define HIP_MATRIX_HPP

#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

/**
 * @brief Enum class defining matrix layout options
 */
enum class matrix_layout
{
    row_major, ///< Row-major layout (elements consecutive in memory by row)
    col_major ///< Column-major layout (elements consecutive in memory by column)
};

// Enum to specify which matrix is being accessed (A or B)
enum class matrix_input
{
    matrix_a,
    matrix_b
};

/**
 * @brief Template class representing a matrix with configurable layout
 * @tparam T Data type of matrix elements
 * @tparam Layout Matrix memory layout (row_major or col_major)
 */
template<class T, matrix_layout Layout = matrix_layout::row_major>
class matrix
{
public:
    using value_type                      = T;
    static constexpr matrix_layout layout = Layout;

    /**
     * @brief Construct a matrix with specified dimensions
     * @param m Dimension of m
     * @param n Dimension of n
     */
    matrix(size_t m, size_t n) : m_(m), n_(n), data_(m * n)
    {
        if(m == 0 || n == 0)
        {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
    }

    // Prevent copying to avoid accidental data transfers
    matrix(const matrix&)            = delete;
    matrix& operator=(const matrix&) = delete;

    // Allow moving
    matrix(matrix&&)            = default;
    matrix& operator=(matrix&&) = default;

    /**
     * @brief Get element at specified position
     * @param i Row index
     * @param j Column index
     * @return Reference to element
     */
    T& operator()(size_t i, size_t j)
    {
        return data_[get_index(i, j)];
    }

    /**
     * @brief Get element at specified position (const version)
     * @param i Row index
     * @param j Column index
     * @return Const reference to element
     */
    const T& operator()(size_t i, size_t j) const
    {
        return data_[get_index(i, j)];
    }

    /**
     * @brief Get raw pointer to matrix data
     * @return Pointer to first element
     */
    T* data()
    {
        return data_.data();
    }
    const T* data() const
    {
        return data_.data();
    }

    /**
     * @brief Get number of m in matrix
     * @return Number of m
     */
    size_t m() const
    {
        return m_;
    }

    /**
     * @brief Get number of columns in matrix
     * @return Number of columns
     */
    size_t n() const
    {
        return n_;
    }

    /**
     * @brief Get total size of matrix in elements
     * @return Total number of elements
     */
    size_t size() const
    {
        return m_ * n_;
    }

    /**
     * @brief Set the data pointer and take ownership
     * @param ptr Pointer to data
     */
    void set_data(std::vector<T>& ptr)
    {
        data_ = ptr;
    }

private:
    /**
     * @brief Calculate linear index based on matrix layout
     * @param i Row index
     * @param j Column index
     * @return Linear index into data array
     */
    size_t get_index(size_t i, size_t j) const
    {
        if constexpr(Layout == matrix_layout::row_major)
        {
            return i * n_ + j;
        }
        else
        {
            return j * m_ + i;
        }
    }

    size_t         m_; ///< Number of m in matrix
    size_t         n_; ///< Number of columns in matrix
    std::vector<T> data_; ///< Pointer to matrix data
};

/**
 * @brief Initialize matrix with random values
 * @tparam T Matrix element type
 * @tparam L Matrix layout
 * @param input Matrix to initialize
 */
template<class T, matrix_layout L>
void init_matrix(matrix<T, L>& input)
{
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    //std::uniform_real_distribution<float> dis(0.1f, 0.2f);
    float tmp[5] = {0.1f, 0.125f, 0.15f, 0.175f, 0.2f};

    int l = 0;
    for(size_t m = 0; m < input.m(); ++m)
    {
        for(size_t n = 0; n < input.n(); ++n)
        {
            input(m, n) = static_cast<T>(tmp[l % 5]);
            l++;
        }
    }
}

#endif // HIP_MATRIX_HPP
