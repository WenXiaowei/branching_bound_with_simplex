class Tree:
    """
    Data-structure mimicking a binary tree.
    """

    def __init__(self, p, info, l=None, r=None):
        self.parent = p
        self.info = info
        self.left = l
        self.right = r

    def __get_nodes__(self, tree):
        if tree is None:
            return []
        return [tree] + self.__get_nodes__(tree.left) + self.__get_nodes__(tree.right)

    def get_all_nodes(self):
        return self.__get_nodes__(self)

    def __leaves__(self, tree):
        if tree is None:
            return []
        if tree.left is None and tree.right is None:
            return [tree]
        return self.__leaves__(tree.left) + self.__leaves__(tree.right)

    def get_leaves_nodes(self):
        return self.__leaves__(self)

    def __pt__(self, tree):
        if tree is not None:
            return f"(node: {tree.info.get_info()}, {self.__pt__(tree.left)}, {self.__pt__(tree.right)})"
        return "_"

    def print_tree(self):
        return self.__pt__(self)

    def __str__(self):
        return f"node: {self.info}, l={self.left}, right={self.right}"

    def __count__(self, tree):
        """
        auxiliary method to count the number of nodes of tree
        :param tree:
        :return:
        """
        if tree is None:
            return 0
        return 1 + self.__count__(tree.left) + self.__count__(tree.right)

    def __len__(self):
        """
        count the number of nodes of the tree
        :return: number of the nodes.
        """
        return self.__count__(self)


if __name__ == "__main__":
    t = Tree(None, 1)
    t.left = Tree(t, 2, l=None, r=None)
    t.right = Tree(t, 3, l=None, r=None)

    # print(t.print_tree())
    [print(node.info) for node in t.get_leaves_nodes()]
