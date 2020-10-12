from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# The training data sets is from:
# https://www.kaggle.com/mrisdal/fake-news/data
# https://www.kaggle.com/therohk/million-headlines


def load_data(real_data_txt: str, fake_data_txt: str):
    # 1 for real data, 0 for fake data
    data_list, data_labels = file_to_lists(real_data_txt, 1, fake_data_txt, 0)

    vectorizer = CountVectorizer()
    data_vectors = (vectorizer.fit_transform(data_list))

    x_train, x_non_train, y_train, y_non_train = \
        train_test_split(data_vectors, data_labels, train_size=0.7)

    x_validation, x_test, y_validation, y_test = \
        train_test_split(x_non_train, y_non_train, train_size=0.15)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def file_to_lists(real_data_txt: str, label1: int,
                  fake_data_txt: str, label2: int):
    data_list = []
    label_list = []
    file = open(real_data_txt, "r")
    lines1 = file.readlines()
    for line in lines1:
        data_list.append(line)
        label_list.append(label1)
    file.close()
    file = open(fake_data_txt, "r")
    lines2 = file.readlines()
    for line in lines2:
        data_list.append(line)
        label_list.append(label2)
    return data_list, label_list


def select_knn_model(real_data_txt: str, fake_data_txt: str):

    x_train, y_train, x_validation, y_validation, \
    x_test, y_test = load_data(real_data_txt, fake_data_txt)

    max_k = 20
    curr_k = 1
    knn_model_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    # get the training, validation accuracies and trained model for different k
    while curr_k <= max_k:
        train_accuracy, val_accuracy, model = knn_model_for_different_k(
                                                        x_train, y_train,
                                                        x_validation,
                                                        y_validation, curr_k)
        knn_model_list.append(model)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        curr_k += 1

    # find which model has highest validation accuracy
    max_val_accuracy = max(val_accuracy_list)
    index = 0
    for score in val_accuracy_list:
        if score == max_val_accuracy:
            break
        index += 1
    best_model = knn_model_list[index]
    best_k = index + 1

    # Find accuracy of test data on model that has highest validation accuracy
    test_data_accuracy = calculate_accuracy(best_model, x_test, y_test)
    # Print out the result
    print_result(train_accuracy_list, val_accuracy_list,
                 test_data_accuracy, best_k)


def knn_model_for_different_k(x_train: list, y_train: list,
                              x_validation: list, y_validation: list, k: int):
    # replace the function train_knn_model to train_knn_model_cosine
    # for using argument metric=‘cosine’ in KNeighborsClassifier
    # model = train_knn_model_cosine(x_train, y_train, k)
    model = train_knn_model(x_train, y_train, k)

    train_accuracy = calculate_accuracy(model, x_train, y_train)
    val_accuracy = calculate_accuracy(model, x_validation, y_validation)
    return train_accuracy, val_accuracy, model


def calculate_accuracy(model: KNeighborsClassifier, x: list, y: list):
    predict = model.predict(x)
    return accuracy_score(y, predict)


def train_knn_model(x: list, y: list, k: int):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x, y)
    return model


def print_result(train_accuracy_list: list,val_accuracy_list: list,
                 test_data_accuracy: float, best_k: int):

    for i in range(len(train_accuracy_list)):
        curr_k = i+1
        train_accuracy = train_accuracy_list[i]
        val_accuracy = val_accuracy_list[i]
        print("k = ",curr_k,"  Training accuracy: "
              ,train_accuracy,"  Validation accuracy: ",val_accuracy)

    print("Best_k: ", best_k, "  Test data accuracy: ", test_data_accuracy)

    k_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.plot(k_list, train_accuracy_list)
    plt.plot(k_list, val_accuracy_list)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()


def train_knn_model_cosine(x: list, y: list, k: int):
    model = KNeighborsClassifier(n_neighbors=k,  metric='cosine')
    model.fit(x, y)
    return model


select_knn_model("real_headlines.txt", "fake_headlines.txt")
