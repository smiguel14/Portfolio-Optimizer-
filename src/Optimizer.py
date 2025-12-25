import numpy as np
import cvxpy as cp


class MeanVarianceOptimizer:
  def __init__(self,risk_aversion: float = 10.0,gross_leverage: float = 1.0,w_max: float = 0.02,turnover_penalty: float = 0.0,factor_neutral: bool = False,solver: str = "OSQP"):
        self.risk_aversion = risk_aversion
        self.gross_leverage = gross_leverage
        self.w_max = w_max
        self.turnover_penalty = turnover_penalty
        self.factor_neutral = factor_neutral
        self.solver = solver
        self.w_prev = None


  def _build_covariance(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma).reshape(-1)
        return np.diag(np.maximum(sigma, 1e-8) ** 2)


  def solve(self,mu: np.ndarray,sigma: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        mu = np.asarray(mu).reshape(-1)
        n = len(mu)
        Sigma = self._build_covariance(sigma)
        w = cp.Variable(n)
        ret_term = mu @ w
        risk_term = 0.5 * self.risk_aversion * cp.quad_form(w, Sigma)
        objective = ret_term - risk_term
        
        constraints = [
            cp.sum(w) == 0.0,    #Dollar Neutrality              
            cp.norm1(w) <= self.gross_leverage, # Leverage Contrainst
            w <= self.w_max, # Position Limit Constraints 
            w >= -self.w_max,   
        ]
        if self.factor_neutral and B is not None:
            B = np.asarray(B)
            constraints.append(B.T @ w == 0) # Factor Neutrality

        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=self.solver, verbose=False)
        if w.value is None:
            raise RuntimeError(f"Optimization failed: {problem.status}")

        w_opt = np.asarray(w.value).reshape(-1)
        self.w_prev = w_opt.copy()
        return w_opt
    


    

  
