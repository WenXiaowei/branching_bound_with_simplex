import numpy as np


class Simplex:
    def __init__(self, reduced_cost, param, term, fixing_cost=None):
        """
        :param reduced_cost: coefficients of the objective function
        :param param: parameters of the objective
        :param term: terms of constraints
        """
        if fixing_cost is None:
            fixing_cost = []
        else:
            if len(fixing_cost) != len(reduced_cost):
                raise ValueError("fixing cost array must have the same length of reduced cost!")

        self.fixing_cost = fixing_cost
        if len(term) != param.shape[0]:
            raise ValueError("Number of terms are different to number of constraints!")

        self.terms = term.copy()
        self.n_reduced_cost = len(reduced_cost)

        self.reduced_cost = np.concatenate((reduced_cost, np.zeros((param.shape[0] + 1))))
        param = np.concatenate((param, np.identity(param.shape[0])), axis=1)
        self.params = np.concatenate((param, np.vstack(term)), axis=1)

        self.in_base = np.array(
            np.arange(start=len(reduced_cost), stop=len(reduced_cost) + param.shape[0], dtype=np.int8))
        # print(self.in_base)

    def __get_pivot_coordinates__(self):
        """
        :return: coordinates (x,y) of the pivot where y is the minimum reduced cost
                meanwhile x is the term which has the minimum ratio
        """
        # taking the minimum rc
        min_rc_index = np.min(np.argwhere(self.reduced_cost < 0))
        # print(f"min_rc_index={min_rc_index}")
        # print(self.reduced_cost[min_rc_index])
        # taking the last column of tableau, which are terms.
        known_terms = self.terms
        # taking the column of the min_rc to compute the ratio
        den = self.params[:, min_rc_index]
        # computing the ratio
        ratio = []
        for i in range(len(known_terms)):
            if den[i] <= 0:
                ratio.append(0)
            else:
                ratio.append(known_terms[i] / den[i])
        ratio = np.asarray(ratio, dtype=np.float32)
        # taking the min ratio value
        if np.all(ratio < 0):
            return None
        # print(f"ratio {ratio}")
        min_ratio = np.min(ratio[ratio > 0])
        # print(f"min_ratio ={min_ratio}")
        # print(f"np.argwhere(ratio == min_ratio) ={np.argwhere(ratio == min_ratio)}")
        return np.min(np.argwhere(ratio == min_ratio)), min_rc_index

    def __pivoting__(self, in_idx, out_idx):
        """
        It performs the pivoting operation with pivot the element in (in_idx, out_idx)
        :param in_idx: row, entry variable
        :param out_idx: column of exiting variable
        :return:
        """
        self.in_base[out_idx] = in_idx
        # print(f"In base: {self.in_base}")

        pivot = self.params[out_idx, in_idx]
        # print(f"pivot {pivot}")
        # print("pivot={}".format(pivot))
        # print(f"ratio {ratio}")

        # for i in range(len(self.reduced_cost)):
        params_copy = self.params.copy()
        # params_copy[out_idx, :] = params_copy[out_idx, :] / pivot
        reduced_cost_copy = self.reduced_cost.copy()
        self.reduced_cost = reduced_cost_copy - (reduced_cost_copy[in_idx] / pivot) * params_copy[out_idx, :]

        # print(f"ratio = {ratio}")
        for i in range(self.params.shape[0]):
            if i != out_idx:
                numerator = params_copy[i, in_idx]
                self.params[i, :] = params_copy[i, :] - (numerator / pivot) * params_copy[out_idx, :]
                # print(self.params[i, :])
            else:

                self.params[i, :] = params_copy[i, :] / pivot
                # print(self.params[i, :])
        # print(self.params)

    def __check_optimality__(self):
        """
        :return: it returns a boolean value which indicates if the current tableau is optimal
        """
        is_optimal = True
        for i in range(len(self.reduced_cost)):
            if self.reduced_cost[i] < 0:
                is_optimal = False

        return is_optimal

    def min(self):
        # if all the relative profits  are greater than or equal to 0, then the current basis is the optimal one
        while not self.__check_optimality__():
            # find the column corresponding to max relative profit.
            # let column j have th max rel. profit: xj will enter the basis
            return_value = self.__get_pivot_coordinates__()
            if return_value is not None:
                # print("Pivot coordinates", return_value)
                out_idx, in_idx = return_value
                # perform the min ratio test only on positive elements to determine which variable will leave the basis.
                # min ratio test:  b_r / a_{rj} = min_i{ b_i / a_{ij}}
                # the index of the min element, r, is the leaving row
                # the variable at index r of the basic variables vector will leave the basis
                # make the identity matrix again:
                # the element at index (r,j) will be the pivot element and row r will be the pivot row
                # divide the r-th row by pivot to make it 1. And subtract c(rth row) from other rows to make them 0,
                # where c is the coefficient required to make that row 0.
                self.__pivoting__(in_idx, out_idx)
                # print("reduced cost", self.reduced_cost)
                # print("params\n", self.params)
                # print("In base", self.in_base)
            else:
                print("The problem is unlimited!")
                return None
            # print("***************************** iterating *****************************")

        return self.__get_results__()

    def __get_results__(self):
        """
        :return: values of in-base variables as a tuple
        """
        # print("reduced cost", self.reduced_cost)
        # print("params\n", self.params)
        # print("In base", self.in_base)

        to_return = []
        # print("The result is:")
        for i in range(len(self.in_base)):
            # (variable_index, value)
            # print(f"X{self.in_base[i] + 1} = {self.params[i, -1]}")
            to_return.append((self.in_base[i], self.params[i, -1]))
        # print("Other variables are equal to zero.")
        return to_return, self.reduced_cost[-1]


def main():
    rc = np.array([-2, -1, -3])
    params = np.array([[2, 1, 2],
                       [1, 1, 0],
                       [1, 1, 2]])
    terms = np.array([3, 1, 2])
    sim = Simplex(rc, params, np.transpose(terms))
    res = sim.min()
    if res is not None:
        variables, best_sol = res
        print("variables: {}".format(variables))
        print("best sol = {}".format(best_sol))


if __name__ == "__main__":
    main()
