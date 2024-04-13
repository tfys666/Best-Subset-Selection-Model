# Model selection: Taking optimal subset regression as an example
## Optimal subset regression
When conducting regression analysis, we may encounter too many independent variables and multicollinearity, or the entire model may be overfitting
- In response to this situation, it is possible to manually screen independent variables that have an impact on the dependent variable based on experience, such as the distance from surrounding schools, to predict housing prices
- But usually we are not experts in the relevant field and do not understand the independent variables that may affect the dependent variable. Therefore, we need to use algorithms to obtain the model with the best prediction performance, and then approach the real model, such as optimal subset regression
- Optimal subset regression, which involves fitting all possible combinations of p predictor variables using least squares regression
- Specifically, for a model with 1 variable, fit $p$ models; For a model with two variables, fit $p(p-1)/2$ models, and so on, for a total of $2^p-1$ models
## Cross validation
-Divide the data into $K$ folds
-Remove the k-th fold data in sequence, train the model, and provide the predicted value for the k-th fold data, where $k=1, \dots, K$
-Build $K$ fold cross validation error
$$\frac 1 n  \sum_{k=1}^K \sum_{i=1}^{n_k} \left(y_{ki} - \hat{y}_{ki}^{(-k)}\right)^2$$
