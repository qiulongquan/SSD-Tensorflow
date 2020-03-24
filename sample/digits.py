# digits.py
from sklearn import svm, datasets
from hyperdash import Experiment

# Preprocess data
digits = datasets.load_digits()
test_cases = 50
X_train, y_train = digits.data[:-test_cases], digits.target[:-test_cases]
X_test, y_test = digits.data[-test_cases:], digits.target[-test_cases:]

# Create an experiment with a model name, then autostart
exp = Experiment("Digits Classifier")
# Record the value of hyperparameter gamma for this experiment
gamma = exp.param("gamma", 0.1)
# Param can record any basic type (Number, Boolean, String)

classifer = svm.SVC(gamma=gamma)
classifer.fit(X_train, y_train)

# Record a numerical performance metric
exp.metric("accuracy", classifer.score(X_test, y_test))

# Cleanup and mark that the experiment successfully completed
exp.end()