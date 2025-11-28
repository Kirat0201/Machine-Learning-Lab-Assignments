import numpy as np
# (a)
# (i)
array = np.array([10,52,62,16,16,54,453])
sorted_array = np.sort(array)
print(sorted_array)
# (ii)  
sorted_indices = np.argsort(array)
print(sorted_indices)
# (iii)
smallest_4 = np.sort(array)[:4]
print(smallest_4)
# (iv)
largest_5 = np.sort(array)[-5:]
print(largest_5)
# (b)
a = np.array([1.0,1.2,2.2,2.0,3.0,2.0])
integer = a == a.astype(int)
integer_element = a[integer]
print(integer_element)
float = a != a.astype(int)
float_element = a[float]
print(float_element)