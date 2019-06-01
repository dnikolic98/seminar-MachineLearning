from statistics import mean
import numpy as np

class LinearRegression:

    def fit(self, xs, ys):
        numerator_m = (mean(xs) * mean(ys) - mean(xs * ys))
        denominator_m = (mean(xs) * mean(xs) - mean(xs*xs))
        self.m = numerator_m / denominator_m
        self.b = mean(ys) - self.m*mean(xs)
        
    def predict(self, predict_x):
        return (self.m * predict_x) + self.b
    
    def squared_error(self, ys_orig, ys_line):
        return sum((ys_line - ys_orig)**2)

    def score(self, xs_test, ys_test):
        y_test_line = [self.predict(x) for x in xs_test]
        y_mean_line = [mean(ys_test) for y in ys_test]
        squared_error_regr = self.squared_error(ys_test, y_test_line)
        squared_error_y_mean = self.squared_error(ys_test, y_mean_line)
        return 1 - squared_error_regr/squared_error_y_mean
