from matchingtools.core import Tensor
from matchingtools.indices import Index
from matchingtools.rules import Rule


kappa, mu, sigma = Tensor.make('kappa', 'mu', 'sigma')
i, j, k, x, y, z = Index.make(*'i j k x y z'.split())

target = kappa(z) * sigma(i, j) * mu(j) + mu(i, j)
rule = Rule(
    sigma(x, y) * mu(y),
    kappa(x, z, z) + sigma(z, z, y)
)

print('{} /. {} -> {} = {}'.format(
    target, rule.pattern, rule.replacement, rule.apply(target)
))
