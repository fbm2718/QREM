from threading import Thread, Lock


class GeneralTensorCalculator:

    def __init__(self, f):
        self.f = f
        self.tensor = dict()
        self.lock = Lock()

    # Calculates tensor for given object using function f provided in the constructor.
    def calculate_tensor(self, objects: list) -> dict:
        self.tensor = dict()

        # Get indices combinations
        indices_combinations = self.__get_indices_combinations(objects.copy())

        # Initialize tensor with empty values
        self.__initialize_tensor(indices_combinations)

        # Prepare for multiprocessing
        threads = []

        # Fill tensor dict with values
        for combination in indices_combinations:
            thread = Thread(target=self.count_tensor_value_for_combination, args=(combination, objects,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.tensor

    # Calculates tensor for given object using function f provided in the constructor, but returns
    # results as a list with increasing order.
    def calculate_tensor_to_increasing_list(self, objects: list) -> list:
        tensor_dict = self.calculate_tensor(objects)
        tensor_list = self.__get_increasing_objects_dict_from_list(tensor_dict)
        return tensor_list

    def __get_increasing_objects_dict_from_list(self, d: dict) -> list:
        objects_list = []
        dict_keys = list(d.keys())

        # This may cause problem if the tensor was somehow calculated for dicts, if it's even possible.
        if type(d[dict_keys[0]]) == dict:
            for k in dict_keys:
                objects_list += self.__get_increasing_objects_dict_from_list(d[k])
        else:
            for k in dict_keys:
                objects_list.append(d[k])

        return objects_list

    # This method has to be public in order to be used in multiprocessing.
    def count_tensor_value_for_combination(self, combination, objects):

        # First count function value for better multiprocessing performance
        f_arguments = []

        # Note that len of objects should be equal to len of current_combination as it contains index value
        # for each object.
        for i in range(len(objects)):
            f_arguments += [objects[i][combination[i]]]

        # Calculate function value now, so that tensor is locked for less time.
        function_value = self.f(f_arguments)

        self.lock.acquire()
        try:
            tensor = self.tensor
            for i in range(len(combination) - 1):
                tensor = tensor[combination[i]]
            tensor[combination[-1]] = function_value
        finally:
            self.lock.release()

    # Get combinations of all possible indices variation.
    def __get_indices_combinations(self, objects: list) -> list:
        # Finish if objects list is empty. This should not happen.
        if len(objects) == 0:
            return None

        combinations = []

        # Termination condition
        # If it's the last element of the list of objects create and return list of it's possible indexes
        if len(objects) == 1:
            for i in range(len(objects[0])):
                combinations.append([i])
            return combinations

        # Recursive call
        # If there are still multiple objects take first and make recursive call
        first_object = objects.pop(0)
        other_objects_combination = self.__get_indices_combinations(objects.copy())

        # Then add to combinations sum of other_objects_combinations and popped object indices
        # Note that order is important here
        for i in range(len(first_object)):
            for combination in other_objects_combination:
                combinations.append([i] + combination)

        return combinations

    # Initializing tensor as dict of dict of dict ... with zeros. Initialization for each combination is performed
    # once at a time.
    def __initialize_tensor(self, indices_combination) -> None:
        # Error check.
        if len(indices_combination) == 0:
            return

        for combination in indices_combination:
            self.__initialize_combination_path(self.tensor, combination.copy())

    # Initializing tensor with 0 for given path
    def __initialize_combination_path(self, tensor, combination):
        # Error check
        if len(combination) == 0:
            return

        # Termination condition
        if len(combination) == 1:
            tensor[combination[0]] = 0
            return

        # Recursive call
        initialized_index = combination.pop(0)
        if not tensor.keys().__contains__(initialized_index):
            tensor[initialized_index] = dict()
        self.__initialize_combination_path(tensor[initialized_index], combination)
