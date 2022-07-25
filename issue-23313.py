import numpy as np
from sklearn.model_selection import train_test_split

data_arr = np.array([1,2915000000,  4890000000, 14145000000,2915000000,  4890000000, 14145000000,2915000000,  4890000000, 14145000000,1])
class_arr = np.array([1,1,2,3,1,2,3,1,2,3,2])
np.unique(data_arr)

x_train, x_test, y_train, y_test = train_test_split(
    data_arr, 
    class_arr, 
    test_size=0.3, 
    random_state=5, 
    shuffle=True, 
    stratify=class_arr
)

print("x_train: ")
print(np.unique(x_train))

print("data_arr: ")
print(np.unique(data_arr))
