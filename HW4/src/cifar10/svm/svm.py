import numpy
import data_handler as dh

def cost_function(W, train_image_set, train_label_set, regularization_strength, delta):
    train_number = train_image_set.shape[0]
    scores = train_image_set.dot(W.T)
    correct_class_scores = scores[range(train_number), train_label_set]
    margins = scores - correct_class_scores[:, numpy.newaxis] + delta
    # margin bounds reset
    margins = numpy.maximum(margins, 0)
    margins[range(train_number), train_label_set] = 0
    loss = numpy.sum(margins) / train_number + 0.5 * regularization_strength * numpy.sum(W * W)

    # gradient descent parameter
    ground_true = numpy.zeros(margins.shape)
    ground_true[margins > 0] = 1
    total_margins = numpy.sum(ground_true, axis=1)
    ground_true[range(train_number), train_label_set] -= total_margins
    gradient = ground_true.T.dot(train_image_set) / train_number + regularization_strength * W

    return loss, gradient


def train(W, train_image_set, train_label_set, learning_rate, regularization_strength, delta, batch_number, iter_number, output):
    train_number = train_image_set.shape[0]
    dim_number = train_image_set.shape[1]
    class_number = numpy.max(train_label_set) + 1

    if W is None:
        W = 0.001 * numpy.random.randn(class_number, dim_number)

    loss_history = []
    for i in range(iter_number):
        sample_index = numpy.random.choice(train_number, batch_number, replace=False)
        train_image_range = train_image_set[sample_index, :]
        train_label_range = train_label_set[sample_index]

        loss, gradient = cost_function(W, train_image_range, train_label_range, regularization_strength, delta)
        loss_history.append(loss)
        W -= learning_rate * gradient

        if output and  i % 100 == 0:
            print('iteration', i, "/", iter_number, ": loss", loss)

    return loss_history, W

def predictor(W, test_image_set):
    scores = test_image_set.dot(W.T)
    predicts = numpy.zeros(test_image_set.shape[0])
    predicts = numpy.argmax(scores, axis=1)
    return predicts


def parameters_optimizer(learning_rate_range, regularization_strengths_range, train_image_set, train_label_set, val_image_set, val_label_set, delta, batch_number, iter_number):
    best_learning_rate = 0
    best_regularization_strength = 0
    best_accuracy = 0

    for i in learning_rate_range:
        for j in regularization_strengths_range:
            print("\noptimizing parameters with learning_rate =", i, ", regularization_strength =", j)
            _, W = train(None, train_image_set, train_label_set, i, j, delta, batch_number, iter_number, True)
            predicts = predictor(W, val_image_set)
            accuracy = numpy.mean(val_label_set == predicts)
            if (best_accuracy < accuracy):
                best_accuracy = accuracy
                best_learning_rate = i
                best_regularization_strength = j

    print("\nlearning_range =", i, ", regularization_strength =", j, " after optimization")
    return best_learning_rate, best_regularization_strength


if __name__ == "__main__":
    # import dataset
    data_set_path = '../dataset/'
    image_set, label_set, test_image_set, test_label_set = dh.load_cifar10(data_set_path)

    # dataset division
    train_number = 49000
    val_number = 1000
    val_range = range(train_number, train_number + val_number)
    val_image_set = image_set[val_range]
    val_label_set = label_set[val_range]
    train_image_set = image_set[:train_number]
    train_label_set = label_set[:train_number]

    train_image_set = numpy.reshape(train_image_set, (train_image_set.shape[0], -1))
    test_image_set = numpy.reshape(test_image_set, (test_image_set.shape[0], -1))
    val_image_set = numpy.reshape(val_image_set, (val_image_set.shape[0], -1))

    mean_image = numpy.mean(train_image_set, axis=0)
    train_image_set = train_image_set - mean_image
    test_image_set = test_image_set - mean_image
    val_image_set = val_image_set - mean_image

    train_image_set = numpy.hstack([train_image_set, numpy.ones((train_image_set.shape[0], 1))])
    test_image_set = numpy.hstack([test_image_set, numpy.ones((test_image_set.shape[0], 1))])
    val_image_set = numpy.hstack([val_image_set, numpy.ones((val_image_set.shape[0], 1))])


    opt_delta = 1
    opt_batch_number = 200
    opt_iter_number = 1000
    delta = 1
    batch_number = 200
    iter_number = 1500
    learning_rate_range = [1e-7, 1e-5]
    regularization_strengths_range = [5e4, 1e5]
    learning_rate, regularization_strength = parameters_optimizer(learning_rate_range, regularization_strengths_range, train_image_set, train_label_set, val_image_set, val_label_set, opt_delta, opt_batch_number, opt_iter_number)
    _, W = train(None, train_image_set, train_label_set, learning_rate, regularization_strength, delta, batch_number, iter_number, True)
    predicts = predictor(W, test_image_set)
    print("\n\nsvm predict accuracy:", numpy.mean(predicts == test_label_set))