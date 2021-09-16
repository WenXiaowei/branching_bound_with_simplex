class Node:
    def __init__(self, var_name, index, upperbound, ad_sol, assigned_value=None, picked_index=None):
        """

        :param var_name:
        :param index:
        :param upperbound:
        :param ad_sol:
        :param assigned_value:
        status: -1 close, 0 normal state, 1 best node
        """
        self.var_name = var_name
        self.upper_bound = upperbound
        self.ad_sol = ad_sol
        self.__is_closed__ = False

        self.status = 0
        if assigned_value is None:
            assigned_value = []
        self.assigned_value = assigned_value
        self.index = index
        if picked_index is None:
            picked_index = []
        self.picked_values = picked_index

    def get_assigned_values(self):
        unzipped = zip(*self.assigned_value)
        unzipped_list = list(unzipped)
        # print(unzipped_list)
        if len(unzipped_list) == 0:
            return [], []
        indexes = list(unzipped_list[0])
        values = list(unzipped_list[1])
        return indexes, values

    def is_closed(self):
        return self.__is_closed__

    def close(self):
        self.__is_closed__ = True
        self.status = -1
        # print(f"Closing node with varname={self.var_name}=0 because not improving")

    def __str__(self):
        return f"varname: {self.var_name}, assigned_value: {self.assigned_value}"

    def get_info(self):
        return f"varname: {self.var_name}, av: {self.assigned_value}, as: {self.ad_sol}, UB: {self.upper_bound}, " \
               f"status: {self.status}, index: {self.index} "

    def __eq__(self, other):
        return type(other) == Node and other.var_name == self.var_name \
               and self.upper_bound == other.upper_bound and self.ad_sol == other.ad_sol

    def __neq__(self, other):
        return not self.__eq__(other)
