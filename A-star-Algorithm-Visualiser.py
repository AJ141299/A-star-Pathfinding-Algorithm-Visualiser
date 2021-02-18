import pygame
import math
from queue import PriorityQueue

WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))  # main surface window
pygame.display.set_caption(
    "A* Path Finding Algorithm Visualiser")  # window title

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Node:
    def __init__(self, row: int, col: int, width: int, total_rows: int):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self) -> tuple:
        """Row and column position of the node.

        Returns:
            row, column (tuple): row and column
        """
        return self.row, self.col

    def is_closed(self) -> bool:
        """Checks whether the node has already been visited.

        Returns:
            bool: True if node is red.
        """
        return self.color == RED

    def is_open(self) -> bool:
        """Checks whether the node has not been visited.

        Returns:
            bool: True if a node is green.
        """
        return self.color == GREEN

    def is_barrier(self) -> bool:
        """Checks whether the node is a barrier (black) node.

        Returns:
            bool: True if a node is black.
        """
        return self.color == BLACK

    def is_start(self) -> bool:
        """Checks whether the node is the starting (orange) node.

        Returns:
            bool: True if a node is orange.
        """
        return self.color == ORANGE

    def is_end(self) -> bool:
        """Checks whether the node is the end (turquoise) node.

        Returns:
            bool: True if a node is turquoise.
        """
        return self.color == TURQUOISE

    def reset(self):
        """Marks the node as empty (white).
        """
        self.color = WHITE

    def make_start(self):
        """Marks the node as starting (orange) node.
        """
        self.color = ORANGE

    def make_closed(self):
        """Marks the node as visited (red).
        """
        self.color = RED

    def make_open(self):
        """Marks the node has yet to be visited (green).
        """
        self.color = GREEN

    def make_barrier(self):
        """Marks the node as a barrier/obstacle (black).
        """
        self.color = BLACK

    def make_end(self):
        """Marks the node as the end node (turquoise).
        """
        self.color = TURQUOISE

    def make_path(self):
        """Marks the node as final path node (purple).
        """
        self.color = PURPLE

    def draw(self, window: pygame.display):
        """Draws (colors) the node on the window.

        Args:
            window (pygame.display): [description]
        """
        pygame.draw.rect(window, self.color,
                         (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid: list):
        """Checks if a node is available in any or all of the four directions of the node and adds them to a new neighbours list.

        Args:
            grid (list): Grid list containing all the nodes.
        """
        self.neighbours = []

        # down
        not_last_row = self.row < self.total_rows - 1
        if not_last_row:
            barrier_down = grid[self.row + 1][self.col].is_barrier()
            if not barrier_down:
                # appends node beneath
                self.neighbours.append(grid[self.row + 1][self.col])

        # up
        not_first_row = self.row > 0
        if not_first_row:
            barrier_up = grid[self.row - 1][self.col].is_barrier()
            if not barrier_up:
                self.neighbours.append(grid[self.row - 1][self.col])

        # right
        not_last_col = self.col < self.total_rows - 1
        if not_last_col:
            barrier_right = grid[self.row][self.col + 1].is_barrier()
            if not barrier_right:
                self.neighbours.append(grid[self.row][self.col + 1])

        # left
        not_first_col = self.col > 0
        if not_first_col:
            barrier_left = grid[self.row][self.col - 1].is_barrier()
            if not barrier_left:
                self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def heuristic(point1: tuple, point2: tuple) -> int:
    """Calculates manhattan's (L) distance as the heuristic between two points.

    Args:
        point1 (tuple): Point 1.
        point2 (tuple): Point 2.

    Returns:
        int: Heuristic (manhattan) distance between the two given points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from: dict, current_node: Node, draw):
    """Traverses back to start node while drawing path.

    Args:
        came_from (dict): [description]
        current_node (Node): [description]
        draw (lambda func): lambda function preloaded with arguments.
    """
    while current_node in came_from:
        current_node = came_from[current_node]
        current_node.make_path()
        draw()


def algorithm(draw, grid: list, start_node: Node, end_node: Node) -> bool:
    """A* Path Finding algorithm.

    Args:
        draw ([lambda func]): lambda function preloaded with arguments.
        grid (list): Grid containing all the nodes.
        start_node (Node): Starting node.
        end_node (Node): End node.

    Returns:
        bool: True if path found, otherwise False.
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start_node))
    came_from = {}  # came from which node

    # assigns all nodes g score as infinity
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start_node] = 0

    f_score = {node: float("inf") for row in grid for node in row}
    # edge case: to avoid marking any path as the shortest path
    f_score[start_node] = heuristic(start_node.get_pos(), end_node.get_pos())

    open_set_hash = {start_node}  # keeps track of items in the priority queue

    while not open_set.empty():
        for event in pygame.event.get():  # cross button was clicked
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]  # get node
        open_set_hash.remove(current)

        if current == end_node:  # path found
            reconstruct_path(came_from, end_node, draw)
            end_node.make_end()
            return True

        for neighbour in current.neighbours:
            # as we have assumed each node is 1 distance apart
            temp_g_score = g_score[current] + 1

            # found a better way to reach this neighbour
            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + \
                    heuristic(neighbour.get_pos(), end_node.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()
        if current != start_node:
            current.make_closed()  # as node already considered

    return False


def make_grid(rows: int, width: int) -> list:
    """Creates and stores all the nodes of the given rows and width in a data structure and returns it.

    Args:
        rows (int): Number of rows.
        width (int): Width of the total surface.

    Returns:
        list: data structure that stores all the nodes.
    """
    columns = rows
    grid = []
    node_width = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(columns):
            node = Node(i, j, node_width, rows)
            grid[i].append(node)

    return grid


def draw_grid_lines(window: pygame.display, rows: int, width: int):
    """Draws the gride lines that separates each node.

    Args:
        window (pygame.display): Main surface window.
        rows (int): Number of rows.
        width (int): Width of the surface.
    """
    node_width = width // rows
    for i in range(rows):
        pygame.draw.line(
            window, GREY, (0, i * node_width), (width, i * node_width))  # draws horizontal lines
        for j in range(rows):
            pygame.draw.line(
                window, GREY, (j * node_width, 0), (j * node_width, width))  # draws vertical lines


def draw(window: pygame.display, grid: list, rows, width):
    """Draws everything on the surface.

    Args:
        window (pygame.display): Main surface window.
        grid (list): Grid list containing all the nodes.
        rows ([type]): Number of rows.
        width ([type]): Width of the surface.
    """
    window.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(window)

    draw_grid_lines(window, rows, width)
    pygame.display.update()


def get_click_pos(pos: pygame.mouse.get_pos, rows: int, width: int) -> tuple:
    """Helper function to figure out which node the mouse clicked on.

    Args:
        pos (pygame.mouse.get_pos): position coordinates of the mouse click.
        rows (int): Number of rows.
        width (int): Width of the surface.

    Returns:
        tuple: row, column
    """
    node_width = width // rows
    y, x = pos

    row = y // node_width
    column = x // node_width

    return row, column


def main(window, width):
    ROWS = 50
    grid = make_grid(ROWS, width)

    start_node = None
    end_node = None

    run = True
    while run:
        draw(window, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # cross button clicked
                run = False

            if pygame.mouse.get_pressed()[0]:  # left mouse click
                pos = pygame.mouse.get_pos()
                # pos of the clicked node
                row, column = get_click_pos(pos, ROWS, width)
                node = grid[row][column]

                if not start_node and node != end_node:
                    start_node = node
                    start_node.make_start()

                elif not end_node and node != start_node:
                    end_node = node
                    end_node.make_end()

                elif node != end_node and node != start_node:
                    node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # right mouse click
                pos = pygame.mouse.get_pos()
                row, column = get_click_pos(pos, ROWS, width)
                node = grid[row][column]
                node.reset()
                if node == start_node:
                    start_node = None
                elif node == end_node:
                    end_node = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start_node and end_node:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)

                    algorithm(lambda: draw(window, grid, ROWS, width),
                              grid, start_node, end_node)

                if event.key == pygame.K_c:  # clear screen and start over
                    start_node = None
                    end_node = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


main(WIN, WIDTH)
