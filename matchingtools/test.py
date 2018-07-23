from core import TensorBuilder
from indices import Index
from rules import Rule
from utils import GeneralDict


sigma = TensorBuilder("sigma")
kappa = TensorBuilder("kappa")
mu = TensorBuilder("mu")

i = Index('i')
j = Index('j')
l = Index('l')
x = Index('x')
y = Index('y')
z = Index('z')

target = kappa(z) * sigma(i, j) * mu(j) + mu(i, j)
rule = Rule(
    sigma(x, y) * mu(y),
    kappa(x, z, z) + sigma(z, z, y)
)

#print([tensor.statistics for tensor in target._to_operator().tensors])
print('{} /. {} -> {} = {}'.format(
    target, rule.pattern, rule.replacement, rule.apply(target)
))
