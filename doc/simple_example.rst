A simple example
================

In this section we will be creating a simple model with symmetry :math:`SU(2)\times U(1)` containing a complex scalar doublet :math:`\phi` (the Higgs) with hypercharge :math:`1/2` and a real scalar triplet :math:`\Xi` with zero hypercharge that couple as:

.. math::
   \mathcal{L}_{int} = - \kappa\Xi^a\phi^\dagger\sigma^a\phi
   - \lambda \Xi^a \Xi^a \phi^\dagger\phi,

where :math:`\kappa` and :math:`\lambda` are a coupling constants and :math:`\sigma^a` are the Pauli matrices. We will then integrate out the heavy scalar :math:`\Xi` to obtain an effective lagrangian which we will finally write in terms of the operators

.. math::
   \mathcal{O}_\phi=(\phi^\dagger\phi)^3,\;
   \mathcal{O}_{\phi 4}=(\phi^\dagger\phi)^2

Creation of the model
---------------------

The basic building blocks of our model are **tensors** and **fields**. Some examples of the tensors we may need are: structure constants of a group, Clebsch-Gordan coefficients, coupling constants (possibly with flavor indices), masses, numeric coefficients... 

For our example, we will need three tensors, the Pauli matrices and the coupling constants::
   
   sigma = TensorBuilder("sigma")
   kappa = TensorBuilder("kappa")
   lambda = TensorBuilder("lambda")

We will also use three fields: the Higgs doublet, its conjugate and the new scalar::
   
   phi = FieldBuilder("phi", 1, boson)
   phic = FieldBuilder("phic", 1, boson)
   Xi = FieldBuilder("Xi", 1, boson)

The second argument of ``FieldBuilder`` represent the energy dimensions of the field, and the third corresponds its the statistics and can either be ``boson`` or ```fermion``.

Now we are ready to write the interaction lagrangian. To create a sum of operators we use the function ``OpSum`` and pass as arguments the operators to be added together. To create an operator, the function ``Op`` should be used. Its arguments are the tensors and fields that form the operator. Each tensor is called as a function with some non-negative integer arguments, the index indentifiers. Repeated integer index identifiers means contraction of the corresponding indices. In our example::
  
   interaction_lagrangian = -OpSum(
        Op(kappa(), Xi(0), phic(1), sigma(0, 1, 2), phi(2)),
	Op(lambda(), Xi(0), Xi(0), phic(1), phi(1)))

Notice the minus sign in front of ``OpSum``. Minus signs are defined in the library for operators and operator sums. Multiplication ``*`` is defined for operators too (as the concatenation of the tensors they contain).

Integration
-----------

Before doing the integration of the heavy fields, we must specify who they are. 
The heavy fields should be objects of any of the classes: ``RealScalar``, ``ComplexScalar``, ``RealVector``, ``ComplexVector``, ``VectorLikeFermion`` and ``MajoranaFermion``.

If we wanted to integrate out the heavy :math:`\Xi` we would do first::
  
  heavy_Xi = RealScalar("Xi", 1)

Now it is ready to be integrated out. Using the function ``integrate``::

  heavy_fields = [heavy_Xi1]
  max_dim = 6
  effective_lagrangian = integrate(
      heavy_fields, interaction_lagrangian, max_dim)

where ``heavy_fields`` is a list of objects of the classes for heavy fields and ``max_dim`` is the maximum dimension allowed for the operators in the effective lagrangian.

Transformations of the effective lagrangian
-------------------------------------------

The results of the integration usually involve operators that aren't independent. We can use rules to transform them and write the final result in a basis of the space of operators. A rule is a transformation to be applied to operators to obtain an equivalent operator sum.

In our case, after the integration we get operators that contain :math:`(\phi^\dagger\sigma^a\phi)(\phi^\dagger\sigma^a\phi)`. This product can be rewritten in terms of the operator :math:`(\phi^\dagger\phi)^2`. To do this, we can use the :math:`SU(2)` Fierz identity:

.. math::
   \sigma^a_{ij}\sigma^a_{kl}=2\delta_{il}\delta_{kj}-\delta_{ij}\delta_{kl}.

We can define a rule to transform everything that matches the left-hand side of the equality into the right-hand with the code::

  fierz_rule = (
      Op(sigma(0, -1, -2), sigma(0, -3, -4)),
      OpSum(number_op(2) * Op(delta(-1, -4), delta(-3, -2)),
            -Op(delta(-1, -2), delta(-3, -4))))
	      
Notice the use of the function ``number_op``. To introduce a (complex) number ``x`` as a coefficient to an operator, the function ``number_op`` can be used as ``number_op(x) * Op(...)``.

Observe also the appearance of negative indices to represent free (not contracted) indpices. In a rule (given as a tuple with two elements), the free indices in the first element (the pattern) should be 

We should now define the basis of operators in which we want to express the effective lagrangian. We may use the function ``tensor_op`` to create operators with only one tensor inside them as::

  Ophi = tensor_op("Ophi")
  Ophi4 = tensor_op("Ophi4")

and then use some rules to express them in terms of the fields and tensors that appear in the effective lagrangian::

  definition_rules = [
      (Op(phic(0), phi(0), phic(1), phi(1), phic(2), phi(2)),
       OpSum(Ophi)),
      (Op(phic(0), phi(0), phic(1), phi(1)),
       OpSum(Ophi4))]

To apply the Fierz identity to every operator until we get to the chosen basis, we do::

  rules = [fierz_rule] + definition_rules
  final_op_names = ["Ophi", "Ophi4"]
  max_iterations = 2
  transf_eff_lag = apply_rules_until(
      effective_lagrangian, rules, final_op_names, max_iterations)

Output
------

The class ``Writer`` can be used to represent the coefficients of the operators of a lagrangian as plain text and write it to a file::

  eff_lag_writer = Writer(trasnf_eff_lag, final_op_names)
  eff_lag_writer.write_text_file("Xi_example")

It can also to write a LaTeX file with the representation of these coefficients and export it to pdf to show it directly. For this to be done, we should define how the objects that we are using have to be represented in LaTeX code and the symbols we want to be used as indices (in this case empty, as no indices will appear in the coefficients)::

  latex_tensor_reps = {"kappa": r"\kappa", "lambda": r"\lambda"}
  latex_op_reps = {"Ophi": r"\mathcal{O}_{\phi}",
	           "Ophi4": r"\mathcal{O}_{\phi 4}"}
		   
  latex_indices = []
  eff_lag_writer.show_pdf(
      "Xi_example", pdf_viewer, latex_tensor_reps, 
      latex_op_reps, latex_indices)

Where ``pdf_viewer`` is the command-line name of a pdf viewer to show the result.
