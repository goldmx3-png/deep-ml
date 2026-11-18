# Cell 1
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    ans = []
    m,n = len(a[0]),len(a)
    x = len(b)
    if x != m:
        return -1
    for i in range(len(a)):
        dot_product=0
        for j in range(len(b)):
            dot_product = dot_product + a[i][j]*b[j]
        ans.append(dot_product)
    return ans
    pass