class Tree:
    # UNSATISFIABLE = "Constraints are not satisfiable in this node."
    UNSATISFIABLE = -1
    # BEST_SOL = "The current node contains the optimal solution."
    ADMISSIBLE_SOL = 1
    # TO_BE_PROCESSED = "The current node needs to be processed."
    TO_BE_PROCESSED = 0
    PROCESSED = -3
    ROOT_NODE = -2

    def __init__(self, decision_variables, idx, simp, reduced_cost=None, lf=None, rt=None):
        """
        :param decision_variables: decision variables
        :param idx:
        :param simp:
        :param lf:
        :param rt:
        """
        self.decision_variables = decision_variables
        self.reduce_cost = reduced_cost
        self.acceptable = True
        self.left = lf
        self.right = rt
        self.simplex = simp
        self.ad_sol = -1
        self.index = idx
        self.node_state = self.TO_BE_PROCESSED

    def close(self, reason):
        self.acceptable = False
        self.node_state = reason

    def is_closed(self):
        return self.node_state != Tree.TO_BE_PROCESSED

    def set_ad_sol(self, ad_sol):
        self.ad_sol = ad_sol
        self.node_state = self.ADMISSIBLE_SOL

    def get_ad_sol(self):
        if self.ad_sol == -1:
            raise ValueError("The node does not contain a valid solution!")
        return self.ad_sol

    def __list_nodes__(tree):
        if tree is None:
            return []
        toRet = [tree]
        if tree.left is not None:
            toRet += tree.left.__list_nodes__()
        if tree.right is not None:
            toRet += tree.right.__list_nodes__()

        return toRet

    def get_all_nodes(self):
        return self.__list_nodes__()

    def depth(self):
        return self.__get_depth__()

    def __list_leaves__(tree):
        if tree is None:
            return []
        if tree.left is None and tree.right is None:
            return [tree]
        toRet = []
        if tree.left is not None:
            toRet += tree.left.__list_nodes__()
        if tree.right is not None:
            toRet += tree.right.__list_nodes__()
        return toRet

    def get_leaves(self):
        return self.__list_leaves__()

    def get_n_level_nodes(self, n):
        """
        returns  a list of nodes at dept n, if n > max depth of the tree, then an empty list will be returned
        :param n:
        :return:
        """
        return __get_n_level_nodes__(self, n, 0)

    def __get_depth__(tree):
        if tree is None:
            return 0
        return 1 + max(tree.left.__get_depth__(), tree.right.__get_depth__())

    def __str__(self):
        return f"decision variable {self.decision_variables}, ad_sol: {self.ad_sol}"


def __get_n_level_nodes__(tree, n, depth):
    if tree is None:
        return []
    if n == depth:
        return [tree]
    return __get_n_level_nodes__(tree.left, n, depth + 1) + __get_n_level_nodes__(tree.right, n, depth + 1)
