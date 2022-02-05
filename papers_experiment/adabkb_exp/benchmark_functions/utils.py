def flatten_list(lst):
    result = []
    for x in lst:
        if isinstance(x, list):
            for elem in x:
                result.append(elem)
        else:
            result.append(x)
    return result