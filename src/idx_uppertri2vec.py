def idx_uppertri2vec(row, col, n):
    """
    generate the corresponding index when the upper triangular part of a symmetric matrix is enumerated as a vector
    row, col = matrix double indices, with row <= col
    n = number of matrix columns
    """

    assert row <= col

    # return round( n*(row+1) - 0.5*row*(row+1) + col - row )
    return round(n * row - 0.5 * row * (row + 1) + col)
