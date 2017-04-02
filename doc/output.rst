Output of the results
=====================

.. currentmodule:: effective.output

.. note:: This section assumes that the class
   :mod:`effective.output.Writer` that it uses is in the namespace.
   To import it, do::

      from effective.output import Writer
		   
It's usually convenient to organize the final results by presenting
the coefficient to each operator of the effective Lagrangian.
When a set of rules has been applied to the effective Lagrangian so
that it is written as an :class:`effective.operators.OperatorSum` whose
elements are :class:`efttool.operators.Operator` objects each of which
contains one tensor representing the actual operator in the basis and
other tensors representing the coefficient the operator has.

To output the results in this form in a human readable format, the
:class:`Writer` is provided. If ``op_names`` is a list of the names
of the tensors representing the operators in the basis and ``lag`` is
the Lagrangian that we want to write, we do::

  lag_writer = Writer(eff_lag, op_names)

To write the results to a file in plain text, just use::

  lag_writer.write_text_file("filename")

To write it in LaTeX two python dictionaries expressing how the tensors
that appear in the coefficients and how the name of the coefficients
for the operators should be written in LaTeX::

  tensors_latex = {"tensor1": r"latexrep", "tensor2": ..., ...}
  ops_latex = {"op1": r"latexrep", ...}

The values of the dictionary should be code to be written inside some
LaTeX equation environment. It is recommended to use ``r"..."`` instead
of ``"..."`` to easily write instructions as ``\instr`` instead of
the form ``\\instr`` that would be needed for the case with just
``"..."``. The placeholders for the indices should be written in
python's ``str.format`` style ``"{}"``. This implies that whenever
curly braces are needed for the LaTeX code, double braces ``{{...}}``
should be used.

The symbols to de used to represent indices should be given also as
a list of strings containing the LaTeX code representing them::

  indices_latex = ["i", "j", ...]

Finally we can write the LaTeX document using::

  lag_writer.write_latex("filename", tensors_latex, ops_latex, indices_latex)

Or we can instead use :meth:`Writer.show_latex` to write it, compile it
and show it all in method::

  lag_writer.show_latex("filename", pdf_viewer, tensors_latex,
                        ops_latex, indices_latex)
