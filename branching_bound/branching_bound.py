import numpy as np
from simplex.simplex import Simplex
from Tree import Tree


def assign_values(variables, idx):
    """
    Given a list of decision variables, it returns two copies of the list with the list[idx] different, one assigned
    with zero, and the other one assigned with 1.
    :param variables: list of decision variables
    :param idx: index.
    :return: couple of list as described in the above.
    """
    if idx > len(variables):
        raise ValueError("Index out of bound.")
    zero_branch = variables.copy()
    zero_branch[idx] = 0
    one_branch = variables.copy()
    one_branch[idx] = 1
    return zero_branch, one_branch


def compute_simp_with_dv(tableau, rc, rc_pos, variables, pos):
    """
    It applies decision variables over the table based on pos
    :param rc:  reduced cost values
    :param rc_pos: reduced cost position where to use binary decision variable
    :param tableau: tableau table
    :param variables: list of decision variables
    :param pos: decision variables positions
    :return: new tableau based on decision variables.
    """
    tableau_copy = tableau.copy()
    rc_copy = rc.copy()
    for idx in range(len(variables)):
        if idx < len(rc_pos):
            rc_copy[rc_pos[idx]] = rc_copy[rc_pos[idx]] * variables[idx]

        else:
            p_i = pos[idx - len(rc_pos)]
            tableau_copy[p_i[0], p_i[1]] = tableau_copy[p_i[0], p_i[1]] * variables[idx]

    return rc_copy, tableau_copy


def check_satisfiability(variables, constraints, decided_index):
    """
    functions checking whether constraints are all satisfied by variables values.
    :param variables: list of variables
    :param constraints: list con constraints, each constraint should be expressed as a lambda with a single variable,
    and should return a Boolean value.
    :param decided_index: index of already decided decision variables
    :return: if all constraints are satisfiable, True, False otherwise.
    """
    if len(constraints) == 0:
        return True

    satisfiable = True
    i = 0
    last_idx = 0
    while i < decided_index and satisfiable and last_idx <= decided_index and i < len(constraints):
        sat, last_idx = constraints[i](variables)
        if not sat:
            satisfiable = False
        i += 1

    return satisfiable


def get_best_decision_variables(dvs, dv_p, rc_var_pos, constraints, reduced_cost, original_params, terms):
    """
    :param rc_var_pos: positions where to place decision variables in reduced cost
    :param dvs: decision variables, list of binary values
    :param dv_p: list of tuples [(xi,yi)], indicating i-th decision variable
    :param constraints: list of constraints, defined as in the main function
    :param reduced_cost: list of reduced cost for the simplex
    :param original_params: tableau
    :param terms: noticed terms of the tableau
    :return: the list of decision variables that produce the best solution.
    """
    if len(dvs) != len(dv_p) + len(rc_var_pos):
        raise ValueError("Number of decision variables is different to number of positions")

    root_node = Tree(decision_variables=None, idx=None, simp=None)
    root_node.close(reason=Tree.ROOT_NODE)

    zero, one = assign_values(dvs, 0)
    opened_nodes = []
    if check_satisfiability(zero, constraints, 1):
        rc_zero, tableau = compute_simp_with_dv(tableau=original_params.copy(), rc=reduced_cost.copy(),
                                                rc_pos=rc_var_pos,
                                                variables=zero, pos=dv_p)
        left_simp = Simplex(rc_zero, tableau, terms)
        left_res = left_simp.min()
        root_node.left = Tree(decision_variables=zero, reduced_cost=rc_zero, idx=0, simp=left_simp)
        opened_nodes.append(root_node.left)
        if left_res is None:
            root_node.left.close(reason=Tree.PROCESSED)
        else:
            root_node.left.set_ad_sol(left_res[1])

    if check_satisfiability(one, constraints, 1):
        rc_one, tableau = compute_simp_with_dv(tableau=original_params.copy(), rc=reduced_cost.copy(),
                                               rc_pos=rc_var_pos,
                                               variables=one, pos=dv_p)
        right_simp = Simplex(rc_one, tableau, terms)
        right_res = right_simp.min()
        root_node.right = Tree(decision_variables=one, reduced_cost=rc_one, idx=0, simp=right_simp)
        opened_nodes.append(root_node.right)
        if right_res is None:
            root_node.right.close(reason=Tree.UNSATISFIABLE)
        else:
            root_node.right.set_ad_sol(right_res[1])
            root_node.right.close(Tree.PROCESSED)

    # print(opened_nodes)
    current_index = 1
    all_nodes = opened_nodes.copy()
    while current_index < len(dvs):
        """
        for each idx-level, we should build the sub-tree with idx-level+1, so the decided variable +1, 
        then close all not improving nodes, and 
        """
        new_nodes = []
        for processing_node in opened_nodes:
            zero, one = assign_values(processing_node.decision_variables, current_index)
            if check_satisfiability(zero, constraints, current_index):
                rc, params = compute_simp_with_dv(original_params, reduced_cost,
                                                  rc_var_pos, zero, dv_p)
                left_simp = Simplex(rc, params, terms)
                left_res = left_simp.min()

                root_node.left = Tree(decision_variables=zero, idx=current_index, reduced_cost=rc,
                                      simp=left_simp)
                new_nodes.append(root_node.left)
                if left_res is None:
                    root_node.left.close(reason=Tree.UNSATISFIABLE)
                if current_index == len(dvs) - 1:
                    root_node.left.set_ad_sol(left_res[1])
                    root_node.left.close(reason=Tree.ADMISSIBLE_SOL)
                # print(f"adsol zero {ad_sol_left}")

            else:
                print(f"Decision variable {zero} is NOT satisfying all constraints")
                root_node.left = Tree(decision_variables=zero, idx=current_index, simp=None)
                root_node.left.close(reason=Tree.UNSATISFIABLE)

            if check_satisfiability(one, constraints, current_index):
                rc_left, params = compute_simp_with_dv(original_params, reduced_cost,
                                                       rc_var_pos, one, dv_p)
                right_simp = Simplex(rc_left, params, terms)
                root_node.right = Tree(decision_variables=one, idx=current_index, reduced_cost=rc_left,
                                       simp=right_simp)
                new_nodes.append(root_node.right)

                ad_sol_right = right_simp.min()
                if ad_sol_right is None:
                    root_node.right.close(reason=Tree.UNSATISFIABLE)

                if current_index == len(dvs) - 1:
                    root_node.right.set_ad_sol(ad_sol_right[1])
                    root_node.left.close(reason=Tree.ADMISSIBLE_SOL)
                # print(f"adsol zero {ad_sol_left}")
            else:
                print(f"Decision variable {one} is NOT satisfying all constraints")
                root_node.right = Tree(decision_variables=one, idx=current_index, simp=None)
                root_node.right.close(reason=Tree.UNSATISFIABLE)
            # print("iterating")
        opened_nodes = new_nodes
        all_nodes += new_nodes

        # opened_nodes = root_node.get_n_level_nodes(current_index)
        current_index += 1
    print(f"Number of nodes constructed: {len(all_nodes)}")
    # picking the best admissible solution
    # print("all nodes, ", root_node.get_all_nodes())
    # print("All ad_sol", [node.ad_sol for node in all_nodes if node.index == len(dvs) - 1])
    last_level_nodes = [node for node in all_nodes if
                        node.index == len(dvs) - 1 and node.node_state == Tree.ADMISSIBLE_SOL]
    # [print("Reduced cost", node.decision_variables) for node in last_level_nodes]

    if len([node.ad_sol for node in last_level_nodes if node.node_state == Tree.ADMISSIBLE_SOL]) == 0:
        print("Error, there maybe two constraints which are contradicting each other!")
        return None
    best_solution = np.min([node.ad_sol for node in last_level_nodes if node.node_state == Tree.ADMISSIBLE_SOL])
    node = [node for node in last_level_nodes if node.ad_sol == best_solution][0]
    print(f"Best node simplex solution: {node.simplex.min()}")
    print(f"Best node ad_sol: {node.ad_sol}")
    print(f"Best node dvs: {node.decision_variables}")

    simp_sol, _ = node.simplex.min()
    return best_solution, node.decision_variables, simp_sol


def check_sat(constraints, dvs):
    """
    Given a list of constraints as lambda checks whether the assignments in dvs satisfies the constraints.
    :param constraints: list of lambda's as constraints
    :param dvs: list of decision variables
    :return:  True iff the dvs satisfies all constraints, otherwise false.
    """
    # print(f"dvs={dvs}")
    for cons in constraints:
        sat, indx = cons(dvs)
        # print(sat)
        if not sat:
            return False
    return True


def main():
    # Number of DVs
    number = 5
    dv = np.zeros(number)
    # position of decision variables in reduced cost
    rc_pos = [0, 2]
    # position of decision variables in the tableau
    dv_pos = [(0, 1), (0, 1), (2, 2)]

    # constraints, each constraint is a lambda with a list of binary decision variables,
    # and it returns whether the constraints are satisfiable and the biggest index of the decision variable involved
    dv_constraints = [
        # lambda variables: (variables[0] == 1, 0),  # x0 == 0
        # warning: this constraints cannot exist with the previous at the same time
        lambda variables: (variables[0] + variables[1] >= 1, 1),  # x0 + x1 >= 1
        lambda variables: (variables[0] + variables[2] >= 1, 1),  # x0 + x2 >= 1
        lambda variables: (variables[3] + variables[2] == 2, 3),  # x3 + x2 == 2
        lambda variables: (variables[3] >= 1, 3),  # x3 >= 1
        lambda variables: (variables[4] + variables[2] <= 1, 4)  # x4 + x2 <= 1
    ]

    # dv_constraints = [
    #     lambda variables: (variables[0] == 1, 0),  # x0 == 1
    #     lambda variables: (variables[1] == 1, 1),  # x1 == 1
    #     lambda variables: (variables[2] == 1, 2),  # x2 == 1
    #     lambda variables: (variables[3] == 1, 3),  # x3 == 1
    #     lambda variables: (variables[4] == 1, 4),  # x4 == 1
    # ]
    rc = np.array([-2, -1, -3])
    p = np.array([[2, 1, 2],
                  [1, 1, 0],
                  [1, 1, 2]])
    t = np.array([3, 1, 2])

    """
    Minimizing this tableau with binary decision variable. 
        -2(x0)  -1      -3(x1)  0   0   0   0
         2(x2)   1(x3)   1      1   0   0   3
         1       2       3      0   1   0   1 
         2       2       1(x4)  0   0   1   2
    """
    res = get_best_decision_variables(dv, dv_pos, rc_pos, dv_constraints, rc, p, np.transpose(t))
    if res is not None:
        bs, values, simp_sol = res
        print("############################# Solution #############################")
        print(f"Optimum value = {bs}, with:\n"
              f"decision variable = {values}\n"
              f"simplex Solution = {simp_sol}")
        print(check_sat(constraints=dv_constraints, dvs=values))
        print("####################################################################")


if __name__ == "__main__":
    main()
