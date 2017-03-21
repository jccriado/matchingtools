
An example with a heavy vector and a heavy fermion
==================================================

Here we will illustrate more features of the library as the use of covariant derivatives, or the management of Lorentz and flavor indices.

The model will now be the Standard Model coupled to a family of color-singlet vector :math:`\mathcal{L}^i_\mu` with representation :math:`2_{1/2}` under :math:`SU(2)\times U(1)` and another family of vector-like leptons :math:`E^i` with the same representation as the right-handed electron :math:`e_R` of the Standard Model. The interaction lagrangian will be:

.. math::
    \mathcal{L}_{int} = 
	- \gamma_i \mathcal{L}_\mu^{\dagger i} D^\mu \phi
	- \gamma_i (D^\mu\phi)^\dagger\mathcal{L}^i_\mu
	- (\lambda^E_l)_{ij} \bar{E}^i_R \phi^\dagger l^j_L
	  
	- (\lambda^E_l)^*_{ij} \bar{l}^j_L \phi E^i_R
	- z^E_{ijk} \bar{E}^i_{L} \mathcal{L}^{j\dagger}_\mu \gamma^\mu l^k_L
	- z^{E*}_{ijk} \bar{l}^k_{L} \mathcal{L}^j_\mu \gamma^\mu E^i_L

Creation of the model
---------------------

As before, we first create the tensors::

  gamma = TensorBuilder("gamma")
  gammac = TensorBuilder("gammac")

  lambdaEl = TensorBuilder("lambdaElc")
  lambdaElc = TensorBuilder("lambdaElc")

  zE = TensorBuilder("zE")
  zEc = TensorBuilder("zEc")

  sigmaSU2 = TensorBuilder("sigmaSU2")

and the fields::

  phi = TensorBuilder("phi", 1, boson)
  phic = TensorBuilder("phic", 1, boson)

  lL = TensorBuilder("lL", 1.5, fermion)
  lLc = TensorBuilder("lLc", 1.5, fermion)

  L = TensorBuilder("L", 1, boson)
  Lc = TensorBuilder("Lc", 1, boson)

  EL = TensorBuilder("EL", 1.5, fermion)
  ER = TensorBuilder("ER", 1.5, fermion)
  ELc = TensorBuilder("ELc", 1.5, fermion)
  ERc = TensorBuilder("ERc", 1.5, fermion)


For every two-component spinor we need to define two fields: one is the conjugate of the other. Therefore, a Dirac spinor requires four ``FieldBuider`` definitions. The integration part of the library is prepared to deal with spinor in two-component formalism. The tensor builders ``epsUp``, ``epsUpDot``, ``epsDown``, ``epsDownDot``, ``sigma4`` and ``sigma4bar``, corresponding to the tensors :math:`\epsilon^{\alpha\beta}`, :math:`\epsilon^{\dot{\alpha}\dot{\beta}}`, :math:`\epsilon_{\alpha\beta}`, :math:`\epsilon_{\dot{\alpha}\dot{\beta}}`, :math:`\sigma^\mu_{\alpha\dot{\beta}}` and :math:`\sigma_\mu^{\dot{\alpha}\beta}$` are already defined in the library.

The next step now is defining the lagrangian. To introduce covariant derivatives of a tensor, the function ``D`` should be used. Its first argument is its Lorenz index and the second is the tensor for which the derivative is to be taken::

  interaction_lagrangian = -OpSum(
      Op(gamma(0), Lc(1, 2, 0), D(1, phi(2))),
      Op(gammac(0), D(1, phic(2)), L(1, 2, 0)),
	
      Op(lambdaEl(0, 1), ERc(2, 0), phic(3), lL(2, 3, 1)),
      Op(lambdaElc(0, 1), lLc(2, 3, 1), phi(3), ER(2, 0)),
	
      Op(zE(0, 1, 2), ELc(3, 0), Lc(4, 5, 1), 
         sigma4bar(4, 3, 6), lL(6, 5, 2)),
      Op(zEc(0, 1, 2), lLc(3, 4, 2), L(5, 4, 1),
         sigma4bar(5, 3, 6), EL(6, 0)))

Integration
-----------

The heavy fields are::

  heavy_L = ComplexVector("L", "Lc", 3)
  heavy_E = VectorLikeFermion("E", "EL", "ER", "ELc", "ERc", 2)
  heavy_fields = [heavy_L, heavy_E]

The first argument to the constructors of the real boson classes is the name of the corresponding field and the second argument is the number of indices this field carries. For the complex boson and Majorana fermion cases the first two arguments correspond, respectively, to the field and its conjugate, whereas the third one is the number of indices. For vector-like fermions, the args are, in order: the name of the fermion, the names of the fields representing its left-handed, right-handed, conjugate left-handed and conjugate right-handed parts, and the number of indices.

To integrate them out to get an effective lagrangian with operators up to dimension 6, we do::

  effective_lagrangian = integrate(heavy_fields, interaction_lagrangian, 6)

Transformations of the effective lagrangian
-------------------------------------------

Let's say we are interested in the mixed contribution of :math:`L_\mu` and :math:`E`. We can collect the corresponding terms using the fact that they will contain the masses of both heavy fields. The masses are automatically given the name ``"M{name}"`` where ``{name}`` is the name of the field::

  mixed_eff_lag = OpSum(*[op for op in mixed_eff_lag.operators
                          if (op.contains_symbol("ML") and
			      op.contains_symbol("ME"))])

The function ``Operator.contains`` checks whether the tensor name passed to it appears in the operator. Masses are represented by a special kind of tensor, a symbol. They are identified by their name beginning with ``"{"``, ending with ``"}"`` and containing one ``"^"``. This identifiers are used to tell the library to treat this tensor as some power of a constant and collect and multiply its ocurrences inside an operator. The equivalent to ``Operator.contains`` for symbols is ``Operator.contains_symbol``, which we have used above.

The operators corresponding to the mixed contributiones that appear after integration are :math:`(\bar{l}^i_L \gamma^\mu D_\mu\phi)(\phi^\dagger l^j_L)` and it conjugate. Suppose that we want to write the final result in terms of the operators`

.. math::
   \left(\mathcal{O}^{(1)}_{\phi l}\right)_{ij} = 
   \bar{l}^i_L \gamma^\mu l^j_L \phi^\dagger D_\mu \phi;
   \;\;\;\;\;\;
   \left(\mathcal{O}^{(3)}_{\phi l}\right)_{ij} = 
   (\bar{l}^i_L \sigma^a \gamma^\mu l^j_L)
   (\phi^\dagger \sigma^a D_\mu \phi);

and their complex conjugates. We would then use the identity

.. math::
   (\bar{l}^i_L \gamma^\mu D_\mu\phi)(\phi^\dagger l^j_L) = 
   \frac{1}{2}\left(
   \bar{l}^i_L \gamma^\mu l^j_L \phi^\dagger D_\mu \phi 
   +(\bar{l}^i_L \sigma^a \gamma^\mu l^j_L)
   (\phi^\dagger \sigma^a D_\mu \phi)\right)

We can substitute everything that matches the left-hand side by the right-hand side and the conjugate of the LHS by the conjugate of the RHS using the rules::

  rules = [
      (# Pattern
       Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), D(2, phi(1)),
          phic(4), lL(3, 4, -2)),
		
       # Replacement
       OpSum(number_op(0.5) * Op(lLc(0, 1, -1), sigma4bar(2, 0, 3),
                                 lL(3, 1, -2), phic(4), D(2, phi(4))),
             number_op(0.5) * Op(lLc(0, 1, -1), sigmaSU2(2, 1, 3),
                                 sigma4bar(4, 0, 5), lL(5, 3, -2),
				 phic(7), sigmaSU2(2, 7, 8),
				 D(4, phi(8))))),
		
      (# Pattern
       Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), phi(1),
          D(2, phic(4)), lL(3, 4, -2)),
		
       # Replacement
       OpSum(number_op(0.5) * Op(lLc(0, 1, -1), sigma4bar(2, 0, 3),
                                 lL(3, 1, -2), D(2, phic(4)), phi(4)),
	     number_op(0.5) * Op(lLc(0, 1, -1), sigmaSU2(2, 1, 3),
	                         sigma4bar(4, 0, 5), lL(5, 3, -2),
				 D(4, phic(7)), sigmaSU2(2, 7, 8),
				 phi(8))))]

We now have to define the basis of operators in which we want the final lagrangian written. We use the function ``flavor_op`` to create a callable such that, when called with several arguments, it returns an operator with a single tensor whose indices are the arguments given::

  O1phil = flavor_op("O1phil")
  O1philc = flavor_op("O1philc")
  O3phil = flavor_op("O3phil")
  O3philc = flavor_op("O3philc")

  definition_rules = [
      (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3),
          lL(3, 1, -2), phic(4), D(2, phi(4))),
       OpSum(O1phil(-1, -2))),
	 
      (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3),
          lL(3, 1, -2), D(2, phic(4)), phi(4)),
       OpSum(O1philc(-1, -2))),
	 
      (Op(lLc(0, 1, -1), sigmaSU2(2, 1, 3),
          sigma4bar(4, 0, 5), lL(5, 3, -2),
	  phic(7), sigmaSU2(2, 7, 8), D(4, phi(8))),
       OpSum(O3phil(-1, -2))),
	 
      (Op(lLc(0, 1, -1), sigmaSU2(2, 1, 3),
          sigma4bar(4, 0, 5), lL(5, 3, -2),
	  D(4, phic(7)), sigmaSU2(2, 7, 8), phi(8))
       OpSum(O3philc(-1, -2)))]

Finally, we just call ``apply_rules_until`` to move the operators the desired base::

  all_rules = rules + definition_rules
  final_op_names = ["O1phil", "O1philc", "O3phil", "O3philc"]
  transf_eff_lag = apply_rules_until(
      effective_lagrangian, all_rules, final_op_names, 2)

Output
------

We can do as in the previous example to write a text file with the results::

  eff_lag_writer = Writer(trasnf_eff_lag, final_op_names)
  eff_lag_writer.write_text_file("L_E_example")

Now let's specify the LaTeX representation. When there's indices in the tensors we should give the positions where they should appear by using the ``str.format`` -style ``"{}"`` placeholders::

  latex_tensor_reps = {
      "gamma": r"\gamma_{}",
      "gammac": r"\gamma^*_{}",
      "lambdaEl": r"(\lambda^E_l)_{{{}{}}}",
      "lambdaElc": r"(\lambda^E_l)^*_{{{}{}}}",
      "zE": r"z^E_{{{}{}{}}}"
      "zEc": r"z^{{E*}}_{{{}{}{}}}"}

  latex_op_reps = {
      "O1phil": r"(\mathcal{{O}}^{{(1)}}_{{\phi l}})_{{{}{}}}",
      "O1philc": r"(\mathcal{{O}}^{{(1)}}^*_{{\phi l}})_{{{}{}}}",
      "O3phil": r"(\mathcal{{O}}^{{(3)}}_{{\phi l}})_{{{}{}}}",
      "O3philc": r"(\mathcal{{O}}^{{(3)}}^*_{{\phi l}})_{{{}{}}}"}

Then we are ready to show the pdf::

  latex_indices = ["i", "j", "k", "l", "m", "n"]
  eff_lag_writer.show_pdf(
      "L_E_example", pdf_viewer, latex_tensor_reps, 
      latex_op_reps, latex_indices)

