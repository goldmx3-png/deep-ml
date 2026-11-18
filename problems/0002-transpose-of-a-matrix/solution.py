# Cell 1
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    ans = []
    m,n = len(a[0]),len(a)
    # Your code here
    for j in range(m):
        arr = []
        for i in range(n):
            arr.append(a[i][j])
        ans.append(arr)
    return ans
    pass