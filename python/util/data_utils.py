import numpy as np


def get_balanced_indices_sample(y, minlength=None, seed=None):
    if minlength is None:
        minlength = max(y) + 1

    # Map classes to indices of instances.
    class_to_indices = [[] for _ in range(minlength)]
    for index, value in enumerate(y):
        class_to_indices[value].append(index)

    # Find minimum number of instances over all classes.
    n = min([len(indices) for indices in class_to_indices])

    # Sub-sample `n` instances from each class.
    if seed is not None:
        np.random.seed(seed)
    balanced_indices = []
    for value, indices in enumerate(class_to_indices):
        balanced_indices.extend(np.random.choice(indices, n, replace=False))
    return sorted(balanced_indices)


def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_


def get_permutations(A):
    if len(A) == 1:
        return A
    permutations = []
    tail_permutations = get_permutations(A[1:])
    for value in A[0]:
        for tail_permutation in tail_permutations:
            permutations.append([value] + tail_permutation)
    return permutations


def get_majority_indices(Y, minlengths=None, tolerance=0):
    if tolerance >= len(Y):
        raise ValueError('`tolerance` ({:d}) cannot exceed the length of `Y` ({:d}).'.format(tolerance, len(Y)))

    if minlengths is None:
        minlengths = [max(y) + 1 for y in Y]

    # Map instances to their indices.
    instance_to_indices = dict()
    for i in range(Y.shape[1]):
        instance = tuple(Y[:, i])
        if instance not in instance_to_indices:
            instance_to_indices[instance] = []
        instance_to_indices[instance].append(i)

    # Count the totals for each class in each category.
    category_class_counts = [list(np.bincount(y, minlength=minlengths[j])) for j, y in enumerate(Y)]

    # Collect a list of instance indices that fall into all majority classes.
    majority_indices = []
    category_max_indices = [allmax(class_counts) for class_counts in category_class_counts]
    tie_indices = [j for j in range(len(category_max_indices)) if len(category_max_indices[j]) > 1]
    while len(tie_indices) <= tolerance:
        majority_instances = [tuple(p) for p in get_permutations(category_max_indices)]
        found = False
        for majority_instance in majority_instances:
            if majority_instance not in instance_to_indices or len(instance_to_indices[majority_instance]) == 0:
                continue
            # Add the instance to the list of majorities while removing it from the list of candidates.
            index = instance_to_indices[majority_instance].pop()
            majority_indices.append(index)
            # Decrement the counts of that instance's values.
            for j in range(len(majority_instance)):
                value = majority_instance[j]
                category_class_counts[j][value] -= 1
            found = True
            break
        if not found:
            break
        category_max_indices = [allmax(class_counts) for class_counts in category_class_counts]
        tie_indices = [j for j in range(len(category_max_indices)) if len(category_max_indices[j]) > 1]
    return sorted(majority_indices)
