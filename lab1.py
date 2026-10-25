import sympy as sp
from abc import ABC, abstractmethod

class FunctionProvider(ABC):
    def __init__(self, expr_str: str):
        self.expr_str = expr_str

    @abstractmethod
    def evaluate(self, val: float) -> float:
        pass

    @abstractmethod
    def get_derivative_expr(self):
        pass


class SymbolicFunctionProvider(FunctionProvider):
    def __init__(self, expr_str: str, var_name: str = 'x'):
        super().__init__(expr_str)
        self.var_name = var_name
        self.var = sp.Symbol(var_name)
        self.expr = self._parse()

    def _parse(self):
        return sp.sympify(self.expr_str)

    def evaluate(self, val: float) -> float:
        return float(self.expr.subs(self.var, val))

    def get_derivative_expr(self):
        return sp.diff(self.expr, self.var)


class TabularFunctionProvider(FunctionProvider):
    def __init__(self, x_data: list, y_data: list):
        super().__init__("Tabular function")
        self.x_data = x_data
        self.y_data = y_data

    def evaluate(self, val: float) -> float:
        if val in self.x_data:
            return self.y_data[self.x_data.index(val)]
        for i in range(len(self.x_data) - 1):
            if self.x_data[i] <= val <= self.x_data[i+1]:
                x0, x1 = self.x_data[i], self.x_data[i+1]
                y0, y1 = self.y_data[i], self.y_data[i+1]
                return y0 + (val - x0) * (y1 - y0) / (x1 - x0)
        return 0.0

    def get_derivative_expr(self):
        return sp.Float(0.0)


class Shape(ABC):
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color

    @abstractmethod
    def calculate_area(self) -> float:
        pass

    @abstractmethod
    def calculate_perimeter(self) -> float:
        pass

    @abstractmethod
    def intersect(self, other: 'Shape') -> 'Shape':
        pass


class SingleFunctionShape(Shape):
    def __init__(self, name: str, color: str, func_provider: FunctionProvider, x_min: float, x_max: float):
        super().__init__(name, color)
        self.func_provider = func_provider
        self.x_min = x_min
        self.x_max = x_max

    def calculate_area(self) -> float:
        if isinstance(self.func_provider, SymbolicFunctionProvider):
            x = self.func_provider.var
            expr = self.func_provider.expr
            area = sp.integrate(expr, (x, self.x_min, self.x_max))
            return float(sp.N(area))
        else:
            n = 500
            dx = (self.x_max - self.x_min) / n
            total = 0.5 * (self.func_provider.evaluate(self.x_min) + self.func_provider.evaluate(self.x_max))
            for i in range(1, n):
                total += self.func_provider.evaluate(self.x_min + i * dx)
            return float(total * dx)

    def calculate_perimeter(self) -> float:
        base_len = self.x_max - self.x_min
        f_min = self.func_provider.evaluate(self.x_min)
        f_max = self.func_provider.evaluate(self.x_max)
        if isinstance(self.func_provider, SymbolicFunctionProvider):
            x = self.func_provider.var
            deriv = self.func_provider.get_derivative_expr()
            arc_expr = sp.sqrt(1 + deriv**2)
            arc_len = sp.integrate(arc_expr, (x, self.x_min, self.x_max))
            return float(base_len + f_min + f_max + sp.N(arc_len))
        else:
            return float(base_len + f_min + f_max + (base_len**2 + (f_max - f_min)**2)**0.5)

    def intersect(self, other: 'Shape') -> 'Shape':
        return CompositeShape(f"Intersection({self.name}, {other.name})", "purple", [self, other], "intersect")


class DoubleFunctionShape(Shape):
    def __init__(self, name: str, color: str, lower_func: FunctionProvider, upper_func: FunctionProvider, x_min: float, x_max: float):
        super().__init__(name, color)
        self.lower_func = lower_func
        self.upper_func = upper_func
        self.x_min = x_min
        self.x_max = x_max

    def calculate_area(self) -> float:
        if isinstance(self.lower_func, SymbolicFunctionProvider) and isinstance(self.upper_func, SymbolicFunctionProvider):
            x = self.upper_func.var
            diff_expr = self.upper_func.expr - self.lower_func.expr
            area = sp.integrate(diff_expr, (x, self.x_min, self.x_max))
            return float(sp.N(area))
        else:
            n = 500
            dx = (self.x_max - self.x_min) / n
            total = 0.5 * ((self.upper_func.evaluate(self.x_min) - self.lower_func.evaluate(self.x_min)) + 
                           (self.upper_func.evaluate(self.x_max) - self.lower_func.evaluate(self.x_max)))
            for i in range(1, n):
                x_val = self.x_min + i * dx
                total += (self.upper_func.evaluate(x_val) - self.lower_func.evaluate(x_val))
            return float(total * dx)

    def calculate_perimeter(self) -> float:
        base = self.x_max - self.x_min
        h_start = abs(self.upper_func.evaluate(self.x_min) - self.lower_func.evaluate(self.x_min))
        h_end = abs(self.upper_func.evaluate(self.x_max) - self.lower_func.evaluate(self.x_max))
        return float(2 * base + h_start + h_end)

    def intersect(self, other: 'Shape') -> 'Shape':
        return CompositeShape(f"Intersection({self.name}, {other.name})", "orange", [self, other], "intersect")


class CompositeShape(Shape):
    def __init__(self, name: str, color: str, sub_shapes: list, operation: str):
        super().__init__(name, color)
        self.sub_shapes = sub_shapes
        self.operation = operation

    def calculate_area(self) -> float:
        if self.operation == "union":
            return sum(s.calculate_area() for s in self.sub_shapes)
        elif self.operation == "intersect":
            if self.sub_shapes:
                return min(s.calculate_area() for s in self.sub_shapes) * 0.85
        return 0.0

    def calculate_perimeter(self) -> float:
        return sum(s.calculate_perimeter() for s in self.sub_shapes) * 0.95

    def intersect(self, other: 'Shape') -> 'Shape':
        new_sub = self.sub_shapes + [other]
        return CompositeShape(f"Union({self.name})", self.color, new_sub, "union")


class ShapeGraphNode:
    def __init__(self, shape: Shape):
        self.shape = shape
        self.neighbors = []

    def add_neighbor(self, node: 'ShapeGraphNode'):
        if node not in self.neighbors:
            self.neighbors.append(node)

    def get_degree(self) -> int:
        return len(self.neighbors)


class ShapeGraph:
    def __init__(self, is_directed: bool = False):
        self.nodes_map = {}
        self.is_directed = is_directed

    def add_vertex(self, shape: Shape) -> ShapeGraphNode:
        if shape.name not in self.nodes_map:
            node = ShapeGraphNode(shape)
            self.nodes_map[shape.name] = node
        return self.nodes_map[shape.name]

    def add_edge(self, shape1: Shape, shape2: Shape):
        node1 = self.add_vertex(shape1)
        node2 = self.add_vertex(shape2)
        node1.add_neighbor(node2)
        if not self.is_directed:
            node2.add_neighbor(node1)

    def check_connectivity(self) -> bool:
        if not self.nodes_map:
            return True
        start_node = list(self.nodes_map.values())[0]
        visited = set()
        queue = [start_node]
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                for neighbor in curr.neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return len(visited) == len(self.nodes_map)

    def find_path(self, start_name: str, end_name: str) -> list:
        if start_name not in self.nodes_map or end_name not in self.nodes_map:
            return []
        start_node = self.nodes_map[start_name]
        end_node = self.nodes_map[end_name]
        queue = [[start_node]]
        visited = {start_node}
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == end_node:
                return [n.shape.name for n in path]
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return []


if __name__ == "__main__":
    f1_provider = SymbolicFunctionProvider("x**2 + 1", "x")
    f2_provider = SymbolicFunctionProvider("2*x + 3", "x")

    tab_provider = TabularFunctionProvider([0, 1, 2, 3], [1, 3, 2, 5])

    shape1 = SingleFunctionShape("ParabolaShape", "blue", f1_provider, 0.0, 2.0)
    shape2 = SingleFunctionShape("LinearShape", "red", f2_provider, 0.0, 2.0)
    shape3 = DoubleFunctionShape("BetweenCurves", "green", f1_provider, f2_provider, 0.0, 1.0)
    shape4 = SingleFunctionShape("TabularShape", "yellow", tab_provider, 0.0, 3.0)

    print(f"Площа {shape1.name}: {shape1.calculate_area():.4f}, Периметр: {shape1.calculate_perimeter():.4f}")
    print(f"Площа {shape2.name}: {shape2.calculate_area():.4f}, Периметр: {shape2.calculate_perimeter():.4f}")
    print(f"Площа {shape3.name}: {shape3.calculate_area():.4f}, Периметр: {shape3.calculate_perimeter():.4f}")
    print(f"Площа {shape4.name}: {shape4.calculate_area():.4f}, Периметр: {shape4.calculate_perimeter():.4f}")

    intersection_shape = shape1.intersect(shape2)
    print(f"Створено композитну фігуру перетину: {intersection_shape.name}, Площа: {intersection_shape.calculate_area():.4f}")

    shape_graph = ShapeGraph(is_directed=False)
    shape_graph.add_edge(shape1, shape2)
    shape_graph.add_edge(shape2, shape3)
    shape_graph.add_edge(shape3, shape4)

    print("Граф зв'язків фігур зв'язний?", shape_graph.check_connectivity())
    path = shape_graph.find_path("ParabolaShape", "TabularShape")
    print("Шлях між фігурами у графі:", path)