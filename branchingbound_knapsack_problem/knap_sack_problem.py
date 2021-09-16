import numpy as np
from Item import Item
from branchingbound_knapsack_problem.Node import Node
from bbtree import Tree


def get_open_leaves(tree):
    return [node for node in tree.get_leaves_nodes() if not node.info.is_closed()]


def construct_root_node(items, weights):
    picked_values = []
    i = ub = ad_sol = 0
    while i < len(items):
        if weights >= items[i].dimension:
            ub += items[i].utility
            ad_sol += items[i].utility
            weights -= items[i].dimension
            picked_values.append(i)
        else:
            print(f"Found critical variable X{i + 1}")
            break
        i += 1
    ub += np.floor(items[i].utility * (weights / items[i].dimension))
    for j in range(i, len(items)):
        if items[j].dimension <= weights:
            ad_sol += items[j].utility
            weights -= items[j].dimension
            picked_values.append(j)
    return Tree(p=None, info=Node(items[i].name, index=i, upperbound=ub, ad_sol=ad_sol, assigned_value=None,
                                  picked_index=picked_values))


def get_ancestors(node):
    if node is None:
        return []
    ancestors = [node]
    ancestors += get_ancestors(node.parent)
    return ancestors


def compute_ad_sol_and_ub(items, weights, node, index, value):
    """
    Given items, assigned values, and weights, it compute for the assigned values, the maximum admissible solution
    and the upper-bound.
    :param items: items to decide if we want to pick it or not maximizing the sum of utilities
    :param weights: total amount of weight that we can carry
    :param node: the current node of the binary tree we need to develop
    :param index: index of the last critical variable
    :param value: value assigned to the current node,
    :return:
    """
    ub, ad_sol = 0, 0
    picked_items = []
    indexes, values = node.info.get_assigned_values()
    indexes.append(index)
    values.append(value)
    for i in range(len(indexes)):
        if values[i] == 1 and weights > items[indexes[i]].dimension:  # items picked
            picked_items.append(indexes[i])
            ub += items[indexes[i]].utility
            ad_sol += items[indexes[i]].utility
            weights -= items[indexes[i]].dimension
        # else:# value[i] == 0 item discarded
    critical_variable_index = -1

    for i in range(len(items)):
        if i not in indexes:  # items[i] is not already assigned
            if weights > items[i].dimension:  # weights > item[i].dimension, so, we can pick it.
                ub += items[i].utility
                ad_sol += items[i].utility
                weights -= items[i].dimension
                picked_items.append(i)
            else:  # weights < items[i].dimension, so, we cannot pick it up.
                ub += np.floor(items[i].utility * (weights / items[i].dimension))
                critical_variable_index = i
                break
    for j in range(critical_variable_index, len(items)):
        if weights > items[j].dimension:
            ad_sol += items[j].utility
            weights -= items[j].dimension
            picked_items.append(j)
    return critical_variable_index, ad_sol, ub, picked_items


def construct_node_children(items, weight, nodes):
    """
    It constructs or the current tree_node two children with item[index] as critical variable.
    :param items: list of items
    :param weight: total amount of weight
    :param nodes: node or list of nodes to open
    """
    if type(nodes) != list:
        nodes = [nodes]
    for tree_node in nodes:
        index = tree_node.info.index
        tree_node.info.close()
        assigned_v = tree_node.info.assigned_value

        # costruire node sx e dx
        idx, adsol, ub, picked_item = compute_ad_sol_and_ub(items, weight, tree_node, index, 0)
        right_node = Node(var_name=items[idx].name, ad_sol=adsol, index=idx, upperbound=ub,
                          assigned_value=[(index, 0)] + assigned_v, picked_index=picked_item)

        idx, adsol, ub, picked_item = compute_ad_sol_and_ub(items, weight, tree_node, index, 1)
        left_node = Node(var_name=items[idx].name, ad_sol=adsol, index=idx, upperbound=ub,
                         assigned_value=[(index, 1)] + assigned_v, picked_index=picked_item)

        tree_node.right = Tree(p=tree_node, info=right_node)
        tree_node.left = Tree(p=tree_node, info=left_node)


def get_next_node_to_open(open_leaves):
    """
    :param open_leaves: list of nodes or node that is not closed
    :return: it returns a list of nodes or a single node that has the maximum Upper-bound,
     if more nodes has the same maximum UB, then a list of these nodes are returned.
    """
    ubs = [node.info.upper_bound for node in open_leaves if not node.info.is_closed()]
    if len(ubs) == 0:
        return []

    toReturn = []
    max_ubs = [np.max(ubs)]
    for node in open_leaves:
        decided_variables = [node.info.index for node in get_ancestors(node.parent)]
        # print("decided variables", decided_variables)
        if node.info.upper_bound == max_ubs and node.info.index not in decided_variables:
            print(f"Opening node with critical variable X{node.info.index + 1}")
            toReturn.append(node)
    # print("return: ", toReturn)
    return toReturn


def print_solution(node, items):
    indexes = node.info.picked_values
    indexes = sorted(indexes)
    [print(f"Item {items[idx].name}=1,") for idx in indexes]
    print("Other items are not picked")


def build_solution(items, weight):
    items = sorted(items, reverse=True)
    # print(items)
    tree = construct_root_node(items, weight)

    construct_node_children(items, weight, [tree])
    opened_leaves = get_open_leaves(tree)
    current_best_sol = max([node.info.ad_sol for node in tree.get_all_nodes()])
    while len(opened_leaves) >= 1:
        # closing all nodes with upper-bound < current_best_sol
        [node.info.close() for node in opened_leaves if node.info.upper_bound <= current_best_sol]
        # nodes = [node for node in opened_leaves if not node.info.is_closed()]
        nodes = get_open_leaves(tree)

        nodes_to_open = get_next_node_to_open(nodes)
        construct_node_children(items, weight, nodes_to_open)
        if len(nodes_to_open) == 0:
            print("They are no node to open.")
            all_nodes = tree.get_all_nodes()
            best_sol = np.max([node.info.ad_sol for node in all_nodes])
            print(f"Best solution is {best_sol}, with assignments: ")
            for n in all_nodes:
                if n.info.ad_sol == best_sol:
                    print_solution(n, items)
                    break
            break
        current_best_sol = np.max([node.info.ad_sol for node in tree.get_all_nodes()])
    return tree


if __name__ == "__main__":
    item_list = [Item("x1", 24, 10), Item("x2", 21, 10), Item("x3", 12, 6), Item("x4", 21, 12),
                 Item("x5", 33, 24), Item("x6", 18, 18), Item("x7", 9, 10)]
    available_space = 51

    # item_list = [Item("x1", 36, 12), Item("x2", 15, 6), Item("x3", 3, 2),
    #              Item("x4", 5, 3), Item("x5", 11, 5), Item("x6", 30, 9)]
    # available_space = 17

    sol_tree = build_solution(item_list, available_space)
