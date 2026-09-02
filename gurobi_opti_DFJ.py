import numpy as np
import gurobipy as gp
from gurobipy import GRB


def gurobi_ant_dfj(d, c, M, W, time_limit=None):
    """
    Fractional orienteering model with Charnes-Cooper transformation
    and DFJ subtour elimination constraints added lazily.

    Parameters
    ----------
    d : array-like, shape (n, n)
        Travel times/distances between nodes.
    c : array-like, shape (n,)
        Time spent at each node.
    M : list-like
        Mandatory node indices. Should include node 0 if node 0 is mandatory.
    W : float
        Total time limit.
    """

    d = np.array(d, dtype=float)
    c = np.array(c, dtype=float)
    c = np.squeeze(np.asarray(c, dtype=float))

    n, n2 = d.shape
    assert n == n2, "d must be a square matrix"

    M = list(M)

    # Big-M for y = t*x.
    # If sum d_ij y_ij = 1 and y = t*x, then t = 1 / total_travel_time.
    # K must be a valid upper bound on t.
    positive_d = d[d > 0]
    if len(positive_d) == 0:
        raise ValueError("At least one positive travel time is required.")
    K = 1.0 / positive_d.min()

    model = gp.Model()

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    t = model.addVar(lb=0.0, ub=K, vtype=GRB.CONTINUOUS, name="t")

    # ------------------------------------------------------------
    # Basic routing constraints
    # ------------------------------------------------------------

    # No self-loops
    model.addConstrs(x[i, i] == 0 for i in range(n))
    model.addConstrs(y[i, i] == 0 for i in range(n))

    # Flow conservation
    model.addConstrs(
        gp.quicksum(y[i, j] for i in range(n)) ==
        gp.quicksum(y[j, i] for i in range(n))
        for j in range(n)
    )

    # Degree check
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i,j] for j in range(n)) <= 1
        )
    
        model.addConstr(
            gp.quicksum(x[j,i] for j in range(n)) <= 1
        )

    # Mandatory nodes must be visited exactly once:
    # since y = t*x, this corresponds to incoming degree 1 in x.
    model.addConstrs(
        gp.quicksum(y[i, j] for i in range(n)) == t
        for j in M
    )

    # Normalization of denominator after Charnes-Cooper
    model.addConstr(
        gp.quicksum(d[i, j] * y[i, j] for i in range(n) for j in range(n)) == 1
    )

    # Total route duration constraint:
    # travel time + time spent at the arrival node
    model.addConstr(
        gp.quicksum((d[i, j] + c[j]) * y[i, j]
                    for i in range(n) for j in range(n)) <= W * t
    )

    # ------------------------------------------------------------
    # Linking constraints: y_ij = t * x_ij
    # ------------------------------------------------------------

    # If x_ij = 1, then y_ij = t.
    # If x_ij = 0, then y_ij = 0.
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(y[i, j] <= K * x[i, j])
                model.addConstr(y[i, j] <= t)
                model.addConstr(y[i, j] >= t - K * (1 - x[i, j]))

    # Objective:
    # maximize total time spent at visited nodes divided by travel time.
    # After transformation this is maximize sum c_i y_ij.
    model.setObjective(
        gp.quicksum(c[i] * y[i, j] for i in range(n) for j in range(n)),
        GRB.MAXIMIZE
    )

    # ------------------------------------------------------------
    # DFJ lazy constraints
    # ------------------------------------------------------------

    def connected_components_from_solution(selected_arcs):
        """
        Finds weakly connected components among selected nodes.
        For subtour elimination, weak components are sufficient:
        any disconnected cycle not containing the depot will be cut.
        """
        adj = {i: set() for i in range(n)}

        selected_nodes = set()
        for i, j in selected_arcs:
            selected_nodes.add(i)
            selected_nodes.add(j)
            adj[i].add(j)
            adj[j].add(i)

        components = []
        unseen = set(selected_nodes)

        while unseen:
            start = unseen.pop()
            stack = [start]
            comp = {start}

            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v in unseen:
                        unseen.remove(v)
                        comp.add(v)
                        stack.append(v)

            components.append(comp)

        return components

    def dfj_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_sol = model.cbGetSolution(model._x)

            selected_arcs = [
                (i, j)
                for i in range(model._n)
                for j in range(model._n)
                if i != j and x_sol[i, j] > 0.5
            ]

            components = connected_components_from_solution(selected_arcs)

            # Node 0 is treated as depot/start-end node.
            depot = 0

            for S in components:
                if depot not in S and len(S) > 1:
                    # DFJ subtour elimination:
                    # sum_{i in S, j in S} x_ij <= |S| - 1
                    model.cbLazy(
                        gp.quicksum(model._x[i, j]
                                    for i in S for j in S if i != j)
                        <= len(S) - 1
                    )

    # Required for cbLazy.
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = time_limit
    model.params.MIPFocus = 1

    model._x = x
    model._n = n

    model.optimize(dfj_callback)

    #return model, x, y, t
    
    var_names = []
    var_values = []
    vystup = []
    
    if model.SolCount == 0:
        return 0, 999, 999, 0
    
    # konec = False
    # for i in range(n):
    #     if sum(x[i, j].X for j in range(n)) > 1:
    #         konec = True
    
    # if model.SolCount == 0 or konec == True:
    #     return 0, 999, 999, 0
    
    for var in model.getVars():
        if var.X > 0: 
            var_names.append(str(var.varName))
            var_values.append(var.X)
            vystup.append([str(var.varName), var.X])
    
    if model.SolCount == 0:
        return 0, 999, 999, 0
    else:
        sol = [0]
        i = 0
        
        while True:
            successors = [j for j in range(n) if x[i, j].X > 0.5]
        
            if len(successors) != 1:
                raise ValueError(
                    f"Node {i} has {len(successors)} outgoing arcs."
                )
        
            i = successors[0]
            sol.append(i)
        
            if i == 0:
                break
            
        # hotovo = False
        # sol = []

        # i = 0
        # ccc = 0
        # while hotovo == False:
        #     j = 0
        #     while j < n:
        #         if 0.5 < x[i, j].X:
        #             sol.append(i)
        #             i = j
        #             break
        #         j += 1
        #     if i == 0:
        #         hotovo = True
                
                
        
        solution = sol
        gap = model.MIPGap
        objVal = model.objVal
        return solution, objVal , gap, vystup   