def APSyn(emb1, emb2, threshold = 1e-5, p = 0.1):
  
  #check if the two vectors have the same shape
  assert emb1.shape == emb2.shape, 'The two vectors have to be of the same shape'

  # reshape them into 1-D arrays to make sorting easier
  emb1, emb2 = emb1.reshape(-1), emb2.reshape(-1)

  # sort vectors descendingly and reshape them into 2-D arrays with 1 row
  emb1_sorted, emb2_sorted = -np.sort(-emb1).reshape(1, -1), -np.sort(-emb2).reshape(1, -1)
  
  # diff_mat is an nxn array in which the [i, j] element is |emb1[i] - emb2[j]|
  diff_mat = np.abs(emb1_sorted - emb2_sorted)

  # identify the indices where the differece of the two vectors are less than the threshold
  indices = list(zip(*np.where(diff_mat < threshold)))

  result = 0
  for i,j in indices:
    result += 2/(np.float_power(i+1, p) + np.float_power(j+1, p))

  return result
