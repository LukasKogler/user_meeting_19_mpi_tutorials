============================================
Advanced Examples - Auxiliary Maxwell Solver
============================================


We will try to build a good preconditioner for the equation:

.. math::

   \nabla \times (\alpha \nabla \times u) = f

After integrating by parts and adding a small :math:`L^2` regularization, we obtain the weak formulation:

Find :math:`u \in H(\text{curl})` such that

..
      \int_\Omega (\alpha \nabla \times u) \cdot \nabla \times v dx = \int_\Omega f \cdot v

.. math::

   \int_\Omega \alpha \nabla \times u \cdot \nabla \times v + \beta u \cdot v dx = \int_\Omega f \cdot v \quad\quad\forall v\in H(\text{curl})
   

