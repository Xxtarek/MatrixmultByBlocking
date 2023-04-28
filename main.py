import time
import numpy as np
import matplotlib.pyplot as plt

def multiply_with_blocking(A, B, block_size):
    assert A.shape[1] == B.shape[0]    # Ensure the dimensions are compatible for matrix multiplication
    m, n = A.shape  #Determine the dimensions of the matrices
    p, q = B.shape
    # Initialize the result matrix with zeros
    C = np.zeros((m, q))

    # Perform matrix multiplication with blocking
    for i in range(0, m, block_size):
        for j in range(0, q, block_size):
            for k in range(0, n, block_size):
                # Multiply the current block of A with the current block of B
                C[i:i+block_size, j:j+block_size] += np.dot(A[i:i+block_size, k:k+block_size], B[k:k+block_size, j:j+block_size])

    return C
#testing with differnet sizes and blocks
matrix_sizes = [100, 200, 500, 1000]
block_sizes = [10, 20, 50, 100]

for matrix_size in matrix_sizes:
    blocking_sizes = []  #Plot the relationship between blocking size and time for each matrix
    elapsed_times = []

    for block_size in block_sizes:
        # Generate random matrices A and B with the specified size
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)

        # Measure the time taken to perform matrix multiplication with blocking
        start_time = time.time()
        C = multiply_with_blocking(A, B, block_size)
        end_time = time.time()

        # Compute the elapsed time and store the result
        elapsed_time = end_time - start_time
        blocking_sizes.append(block_size)
        elapsed_times.append(elapsed_time)
        print(f"Matrix size: {matrix_size} | Block size: {block_size} | Elapsed time: {elapsed_time:.4f} seconds")

    # Plot the results for this matrix size
    plt.plot(blocking_sizes, elapsed_times, label=f"Matrix size {matrix_size}")

plt.xlabel("Blocking size")
plt.ylabel("Elapsed time (seconds)")
plt.legend()
plt.show()

