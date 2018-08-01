def D(index, tensor):
    """
    Interface for the creation of tensors with derivatives applied.
    """
    return tensor.differentiate(index)


def hc(x):
    return x.conjugate()
