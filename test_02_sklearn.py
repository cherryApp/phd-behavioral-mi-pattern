from sklearn.metrics.pairwise import cosine_similarity

list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list2 = [1, 3, 2, 4, 6, 99, 8, 7, 9, 10]

similarity = cosine_similarity([list1], [list2])

print(similarity)