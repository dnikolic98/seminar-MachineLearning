from collections import Counter
import numpy as np

class kNearestNeighbors:

    def fit(self, data, k=5):
        self.k = k
        self.data = data

    def predict(self, predict_value):
        distances = []
        for group in self.data:
            for features in self.data[group]:
                difference = np.array(features)-np.array(predict_value)
                euclidean_distance = np.linalg.norm(difference)
                distances.append([euclidean_distance, group])
        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        return vote_result

    def score(self, test_data):
        correct = 0
        count = 0
        for group in test_data:
            for features in test_data[group]:
                result = self.predict(features)
                if result == group:
                    correct += 1
                count += 1
        return correct/count
