class Point:
    y = None
    x = None
    angle = None

    def __init__(self, y, x, angle=None):
        self.y = y
        self.x = x
        self.angle = angle

    def __lt__(self, other):
        if self.y > other.y:
            return True
        if self.x < other.x:
            return True
        return False

    def __eq__(self, other):
        if self.y == other.y and self.x == other.x:
            return True
        return False

    def __le__(self, other):
        if self < other or self==other:
            return True
        return False

    def __ne__(self, other):
        return not self==other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def setAngle(self, angle):
        self.angle = angle