class Coordinates:
    def __init__(self, x_1=0, x_2=0, y_1=0, y_2=0):
        self.x_1 = x_1
        self.x_2 = x_2
        self.y_1 = y_1
        self.y_2 = y_2

    def update_y_coordinates(self, image, cols):
        step_y = image.shape[1] / cols
        self.y_1 = self.y_2
        self.y_2 = self.y_2 + step_y

    def update_x_coordinates(self, image, rows):
        step_x = image.shape[0] / rows
        self.x_1 = self.x_2
        self.x_2 = self.x_2 + step_x

    def copy(self, src):
        self.x_1 = src.x_1
        self.x_2 = src.x_2
        self.y_1 = src.y_1
        self.y_2 = src.y_2
