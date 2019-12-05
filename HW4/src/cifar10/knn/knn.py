import numpy
import matplotlib.pyplot as plt
import data_handler as dh

def compute_distance(image_set, test_image_set):
    distances = numpy.zeros((test_image_set.shape[0], image_set.shape[0]))
    distances = numpy.multiply(test_image_set.dot(image_set.T), -2) + numpy.sum(numpy.square(test_image_set), axis=1, keepdims=True) + numpy.sum(numpy.square(image_set), axis=1)
    return distances

def predictor(label_set, distances, k):
    predictor = numpy.zeros(distances.shape[0])
    for i in range(distances.shape[0]):
        nearest = label_set[numpy.argsort(distances[i, :])[:k]]
        predictor[i] = numpy.argmax(numpy.bincount(nearest))
    return predictor

def predict(image_set, label_set, test_image_set, k):
    distances = compute_distance(image_set, test_image_set)
    predict_result = predictor(label_set, distances, k)
    return predict_result

if __name__ == "__main__":
    data_set_path = '../dataset/'
    test_range = 25
    accuracys = numpy.zeros(test_range + 1)

    training_set_num = 10000
    image_set, label_set, test_image_set, test_label_set = dh.load_cifar10(data_set_path)
    t_image_set = image_set[:training_set_num, ::]
    t_image_set = numpy.reshape(t_image_set, (t_image_set.shape[0], -1))
    t_label_set = label_set[:training_set_num]
    testing_set_num = 1000
    t_test_image_set = test_image_set[:testing_set_num, ::]
    t_test_image_set = numpy.reshape(t_test_image_set, (t_test_image_set.shape[0], -1))
    t_test_label_set = test_label_set[:testing_set_num]

    for k in range(1, test_range+1):
        print("\nknn algorithm running with k =", k)
        result = predict(t_image_set, t_label_set, t_test_image_set, k)
        accuracy = numpy.sum(result == t_test_label_set) / float(t_test_image_set.shape[0])
        print("predict accuracy =", accuracy)
        accuracys[k] = accuracy

    accuracys = accuracys[1: test_range+1]
    plt.plot(range(1, test_range+1), accuracys, "b--", linewidth=2, marker='o')
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("k nearest neighbor algorithm accuracy distribution on Ks")
    plt.grid(True)
    plt.xticks(numpy.arange(0, test_range+1, 1))
    plt.show()