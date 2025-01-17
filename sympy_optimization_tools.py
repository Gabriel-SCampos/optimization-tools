import sympy as sp
from sympy import Expr, Matrix, MatrixBase, Symbol, nan


class FunctionUtils:
    def __init__(self, f: Expr):
        self.f = f
        self.variables = sorted(list(f.free_symbols), key=lambda s: str(s))

    def grad(self):
        return sp.derive_by_array(self.f, self.variables)

    def part_deriv(self, x: Symbol):
        df_dx = sp.diff(self.f, x)
        return df_dx
    
    def hessian(self) -> MatrixBase:
        hessian_matrix = sp.hessian(self.f, self.variables)
        return hessian_matrix

    def eigenvalues(self):
        hessian_matrix = self.hessian()
        return hessian_matrix.eigenvals()

    def eigenvectors(self):
        hessian_matrix = self.hessian()
        return hessian_matrix.eigenvects()
    
    def T(self):
        eigenvects = self.eigenvectors()
        return Matrix.hstack(*[vect[2][0].normalized() for vect in eigenvects])

    def critical_points(self) -> list[dict[str, Expr]]:
        part_derivs = []
        for var in self.variables:
            part_deriv = self.part_deriv(var)
            part_derivs.append(part_deriv)

        equations = [Eq(derivative, 0) for derivative in part_derivs]
        critical_points = sp.solve(equations, self.variables, dict=True)
        return critical_points
    
