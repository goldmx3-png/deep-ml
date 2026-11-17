# Cell 1
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    # Return the element-wise sum of vectors 'a' and 'b'.
    # If vectors have different lengths, return -1.
    if len(a)!=len(b):
        return [-1]
    ans = []
    for i,c in enumerate(a):
        for j,d in enumerate(b):
            if i==j:
                ans.append(c+d)
    return ans
    pass