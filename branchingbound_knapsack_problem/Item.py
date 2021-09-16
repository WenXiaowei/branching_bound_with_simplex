
class Item:
    def __init__(self, name, utility, dimension):
        self.utility = utility
        self.name = name
        self.dimension = dimension
        self.ratio = round(utility / dimension, 2)

    def __lt__(self, other):
        return self.ratio < other.ratio

    def __le__(self, other):
        return self.ratio <= other.ratio

    def __gt__(self, other):
        return self.ratio > other.ratio

    def __ge__(self, other):
        return self.ratio >= other.ratio

    def __eq__(self, other):
        return self.utility == other.utility and self.name == other.name and \
               self.dimension == other.dimension and self.ratio == other.ratio

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __str__(self):
        return f"name={self.name}\nutility={self.utility}\ndimension={self.dimension}\nratio={self.ratio}\n"
