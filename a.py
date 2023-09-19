my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
k = 5

# Calculate the middle index
middle_index = len(my_list) // 2
m = k // 2
r = k % 2

# Cut the list in the middle
result = my_list[middle_index - m: middle_index + m + r]

print(result)
