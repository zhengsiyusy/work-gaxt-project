cdef extern from "csum.h":
	int sum(int a, int b)

def pysum(int a, int b):
	return sum(a,b)