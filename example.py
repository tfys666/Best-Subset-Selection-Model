import numpy as np
from eva import OptimalSubsetRegression

x = np.loadtxt("prostate/x.txt", delimiter=",")
y = np.loadtxt("prostate/y.txt", delimiter=",")
index = np.loadtxt("prostate/index.txt", delimiter=",", dtype=bool)
names = np.loadtxt("prostate/names.txt", delimiter=",", dtype=str)

regression = OptimalSubsetRegression(x, y, index, names, K=10)

regression.train()
regression.evaluate()
regression.cross_validate()

best_b_0, best_b_1, best_inds = regression.best_model()
print(f"Best Model: b_0={best_b_0}, b_1={best_b_1}, Variables={best_inds}")
