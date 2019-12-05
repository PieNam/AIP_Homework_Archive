import numpy
import data_handler as dh

class TwoNN(object):
    def __init__(self, input_size, hidden_size, class_number, std=1e-4):
        self.parameters = {}
        self.parameters['W1'] = std * numpy.random.randn(hidden_size, input_size)
        self.parameters['b1'] = numpy.zeros(hidden_size)
        self.parameters['W2'] = std * numpy.random.randn(class_number, hidden_size)
        self.parameters['b2'] = numpy.zeros(class_number)

    def loss_function(self, train_image_set, train_label_set, regularization_strength):
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        sample_number = train_image_set.shape[0]

        # forward passing
        ReLU = lambda x: numpy.maximum(0, x)
        z1 = train_image_set.dot(W1.T) + b1
        a1 = ReLU(z1)
        z2 = a1.dot(W2.T) + b2
        scores = z2

        # predicting, return result
        if train_label_set is None:
            return scores

        # training, computing loss
        exp_scores = numpy.exp(scores - numpy.max(scores, axis=1, keepdims=True))
        pro_scores = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)
        ground_true = numpy.zeros(scores.shape)
        ground_true[range(sample_number), train_label_set] = 1
        loss = -numpy.sum(ground_true * numpy.log(pro_scores)) / sample_number + 0.5 * regularization_strength * (numpy.sum(W1 * W1) + numpy.sum(W2 * W2))

        # backward passing, computing gradient
        gradients = {}
        gradient_z2 = -(ground_true - pro_scores) / sample_number
        gradient_w2 = gradient_z2.T.dot(a1)
        gradient_b2 = numpy.sum(gradient_z2, axis=0)
        gradient_a1 = gradient_z2.dot(W2)
        gradient_z1 = gradient_a1
        gradient_z1[a1 <= 0] = 0
        gradient_w1 = gradient_z1.T.dot(train_image_set)
        gradient_b1 = numpy.sum(gradient_z1, axis=0)

        # regularization
        gradients['W1'] = gradient_w1 + regularization_strength * W1
        gradients['b1'] = gradient_b1
        gradients['W2'] = gradient_w2 + regularization_strength * W2
        gradients['b2'] = gradient_b2

        return loss, gradients

    def train(self, train_image_set, train_label_set, val_image_set, val_label_set, regularization_strength, learning_rate, learning_rate_decay, iterations_per_layer_annealing, epoches_number, batch_size, if_log):
        sample_number = train_image_set.shape[0]
        loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        iterations_per_epoch = max(sample_number / batch_size, 1)
        iterations_number = int(epoches_number * iterations_per_epoch)

        for i in range(iterations_number):
            sample_index = numpy.random.choice(sample_number, batch_size, replace=True)
            train_image_batch = train_image_set[sample_index, :]
            train_label_batch = train_label_set[sample_index]
            
            # computing loss
            loss, gradient = self.loss_function(train_image_batch, train_label_batch, regularization_strength)
            loss_history.append(loss)

            self.parameters['W1'] -= learning_rate * gradient['W1']
            self.parameters['b1'] -= learning_rate * gradient['b1']
            self.parameters['W2'] -= learning_rate * gradient['W2']
            self.parameters['b2'] -= learning_rate * gradient['b2']

            if if_log and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, iterations_number, loss))

            if i % iterations_per_epoch == 0:
                train_accuracy_history.append(numpy.mean(self.predictor(train_image_batch) == train_label_batch))
                val_accuracy_history.append(numpy.mean(self.predictor(val_image_set) == val_label_set))
            
            if i % iterations_per_layer_annealing == 0:
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_accuracy_history': train_accuracy_history,
            'val_accuracy_history': val_accuracy_history
        }

    def predictor(self, test_image_set):
        ReLU = lambda x: numpy.maximum(0, x)
        z1 = test_image_set.dot(self.parameters['W1'].T) + self.parameters['b1']
        a1 = ReLU(z1)
        z2 = a1.dot(self.parameters['W2'].T) + self.parameters['b2']
        score = z2

        predicts = numpy.argmax(score, axis=1)
        return predicts



def parameters_optimizer(train_image_set, train_label_set, val_image_set, val_label_set, input_size, hidden_size, class_number, learning_rate_range, learning_rate_decay_range, regularization_strength_range, iterations_per_layer_annealing_range, epoches_number, batch_size):
    best_net = None
    best_accuracy = 0
    # best_parameters = None

    figure_index = 0
    for learning_rate in learning_rate_range:
        for regularization_strength in regularization_strength_range:
            for learning_rate_decay in learning_rate_decay_range:
                for iterations_per_layer_annealing in iterations_per_layer_annealing_range:
                    figure_index += 1
                    print("\noptimizing parameters......")
                    print("  - learning_rate:", learning_rate, "| regularization_strength:", regularization_strength, "| learning_rate_decay:", learning_rate_decay, "| iterations_per_layer_annealing", iterations_per_layer_annealing)
                    NN = TwoNN(input_size, hidden_size, class_number)
                    NN.train(train_image_set, train_label_set, val_image_set, val_label_set, regularization_strength, learning_rate, learning_rate_decay, iterations_per_layer_annealing, epoches_number, batch_size, False)
                    val_accuracy = (NN.predictor(val_image_set) == val_label_set).mean()
                    print("    - predict accuracy in validation:", val_accuracy)

                    if best_accuracy < val_accuracy:
                        best_accuracy = val_accuracy
                        best_net = NN
                        best_parameters = (learning_rate, regularization_strength, learning_rate_decay, iterations_per_layer_annealing)
    print("\nparameters after optimization:")
    print("  - learning_rate:", best_parameters[0], "| regularization_strength:", best_parameters[1], "| learning_rate_decay:", best_parameters[2], "| iterations_per_layer_annealing", best_parameters[3])
    print("with best predict accuracy in validation:", best_accuracy)
    return best_net


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

    # parameters setting
    input_size = 32 * 32 * 3
    hidden_size = 80
    class_number = 10
    learning_rate_range = [8e-4, 9e-4]
    learning_rate_decay_range = [0.95, 0.97, 0.99]
    regularization_strength_range = [1e-2, 1e-1]
    iterations_per_layer_annealing_range = [400, 500]
    epoches_number = 15
    batch_size = 250

    best_net = parameters_optimizer(train_image_set, train_label_set, val_image_set, val_label_set, input_size, hidden_size, class_number, learning_rate_range, learning_rate_decay_range, regularization_strength_range, iterations_per_layer_annealing_range, epoches_number, batch_size)
    accruacy = numpy.mean(best_net.predictor(test_image_set) == test_label_set)
    print("\n\ntwo-layer neural network accuracy:", accruacy)