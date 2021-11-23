import numpy as np


def count_tensor(vectors, tensor=[]):
    # ASSIGNING TERMINATION CONDITION AT THE BEGINNING
    # Check if single vector was passed
    if len(vectors) == 1:
        # Insert it's elements into target if so and return
        for el in vectors[0]:
            tensor.append(el)
        return tensor

    # If there are more than one vectors For each vector take first vector out of the list
    vec = vectors.pop(0)

    # For each element in this vector
    for el in vec:
        # IMPORTANT
        # Copy before each recurrence call
        vs = vectors.copy()
        # Add this vectors i_index-th element list
        tensor.append([])
        # RECURRENCE CALL
        # Call this method on reduced data set
        count_tensor(vs, tensor[-1])
        # Multiply each obtained element by element from vector vec
        # Side note: I believe this could be performed in a more tricky way by putting el inside newly created list
        # I will provide test it in another method.
        # CAUTION -- following method is also use recurrence
        multiply_each_element(tensor[-1], el)

    # Return the tensor
    return tensor


def multiply_each_element(tensor, multiplier=1):
    # ASSIGNING TERMINATION CONDITION AT THE BEGINNING
    # IF tensor is empty just return (I believe this should not happen)
    if len(tensor) == 0:
        return

    # Check if first element is a list. If not, then just multiply elements
    if not isinstance(tensor[0], list):
        for i in range(len(tensor)):
            tensor[i] = np.kron(multiplier, tensor[i])
        return

    # Otherwise call multiplying function for each element (list) of tensor
    for l in tensor:
        multiply_each_element(l, multiplier)


def count_tensor_smartly(vectors, tensor=[1]):
    # ASSIGNING TERMINATION CONDITION AT THE BEGINNING
    # Check if single vector was passed
    # CAUTION: Note how tensor is initiated. That's the whole trick of the method.
    # Compare with multiply each element approach in CountTensor

    resultant_tensor = []

    if len(vectors) == 1:
        multiplier = tensor[0]
        # Insert it's elements into target if so and return
        for el in vectors[0]:
            resultant_tensor.append(np.kron(multiplier, el))
        return resultant_tensor

    # If we're not multiplying information about multiplier should be deleted
    tensor.clear()

    # If there are more than one vectors For each vector take first vector out of the list
    vec = vectors.pop(0)

    # For each element in this vector
    for el in vec:
        # IMPORTANT
        # Copy before each recurrence call
        vs = vectors.copy()
        # Add this vectors i_index-th element list
        resultant_tensor.append([el])
        # RECURRENCE CALL
        # Call this method on reduced data set
        count_tensor_smartly(vs, resultant_tensor[-1])

    # Return the tensor
    return tensor
