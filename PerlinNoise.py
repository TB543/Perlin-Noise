from random import uniform as _uniform, seed as _seed
from numpy import array as _vector
from numpy.linalg import norm as _norm


class PerlinNoise:
    """
    a class for perlin noise with more useful features:
        --> multidimensional noise
        --> ability to create more complex noise with octaves todo load octaves as octaves not PerlinNoise when loading saved obj
        --> ability to generate a "chunk" of noise (faster than getting individual values) todo
        --> ability to save and load PerlinNoise objects
        --> ability to save PerlinNoise values to save processing power at the cost of ram/drive space
    """

    def __init__(self, seed: float = None, octaves: list = None, scale: float = 255.0, minimum: float = -1.0,
                 maximum: float = 1.0, save_data: bool = False, saved_perlin_obj: str = None):
        """
        creates a PerlinNoise object with attributes of the following parameters
        :param seed: the seed for the perlin noise, if not given a random seed will be generated
        :param octaves: a list of Octave objects that will be added to the noise value, note: modifying scale,
            minimum and maximum value will change the way the noise looks
        :param scale: the scale of the noise, the higher the number the more "zoomed in" the noise is, default value is
            255, note: if the value is 1, zero will be given for every point within the noise due to the nature of how
            perlin noise works
        :param minimum: the minimum value that can be returned from the get_value() method, default value is -1
        :param maximum: the maximum value that can be returned from the get_value() method, default value is 1
        :param save_data: determines if calculated values should be saved (saves processing power at the cost of
            ram/drive space), note: changes to class variables will not update these saved values and will result in
            corrupted noise
        :param saved_perlin_obj: json data representing all of the class attributes as a string, if this is given,
            saved data will automatically be loaded
        """

        # loads save data if given
        if saved_perlin_obj:
            self.load(saved_perlin_obj)

        # sets class variables otherwise
        else:
            self.seed = seed if seed else _uniform(-1, 1)
            self.octaves = octaves if octaves else []
            self.scale = scale
            self.min = minimum
            self.max = maximum
            self.save_data = save_data
            self.saved_data = {}

    def get_value(self, *args: float, smooth: bool = False):
        """
        gets the value of the noise at the given coordinates
        :param args: the coordinates of the point in the noise, the number of args determines the dimensions of the
            noise
        :param smooth: determines if the perlin noise should be blended together more, downside is that it causes
            slightly more time to calculate the value, default is false
        :return: a float representing the value at that point
        """

        # returns value if it has previously been saved
        try:
            return self.saved_data[args]

        # calculates value otherwise
        except KeyError:

            # applies scaling to points given and finds origin point and sets variables
            scaled_points = _vector([point / self.scale for point in args])
            origin_points = _vector([int(point) for point in scaled_points])
            dot_products = []

            # calculates the gradient for every corner
            for delta_corner in range(2 ** len(scaled_points)):

                # finds out which corner to use based on delta corner
                delta_corner = [int(string) for string in list(format(delta_corner, 'b'))]
                [delta_corner.insert(0, 0) for _ in range(len(origin_points) - len(delta_corner))]
                corner = _vector(delta_corner) + origin_points

                # sets the seeded value of the corner
                _seed(corner[0] + self.seed)
                corner_seed = _uniform(-1, 1)
                for dimension in range(1, len(corner)):
                    _seed(corner_seed + corner[dimension])
                    corner_seed = _uniform(-1, 1)

                # dot products gradient and distance vectors
                gradient_vector = _vector([_uniform(-1, 1) for _ in range(len(corner))])
                gradient_vector = gradient_vector / _norm(gradient_vector) if len(corner) > 1 else gradient_vector
                dot_products.append(gradient_vector.dot(_vector([
                    scaled_points[dimension] - corner[dimension] for dimension in range(len(corner))])))

            # interpolates gradients
            index = len(scaled_points) - 1
            while len(dot_products) > 1:
                weight = scaled_points[index] - origin_points[index]
                if smooth:
                    dot_products.append((dot_products[1] - dot_products[0]) * ((weight * (weight * 6 - 15) + 10) * (
                            weight ** 3)) + dot_products[0])
                else:
                    dot_products.append((dot_products[1] - dot_products[0]) * (3.0 - weight * 2.0) * (
                            weight ** 2) + dot_products[0])
                dot_products.pop(0)
                dot_products.pop(0)
                index -= 1 if len(dot_products) / 2 == index else 0

            # applies scaling and octaves
            for octave in self.octaves:
                dot_products[0] += octave.get_value(*args)
            scaled_value = (((dot_products[0] + 1) / 2) * (self.max - self.min)) + self.min

            # saves value if applicable and returns value
            if self.save_data:
                self.saved_data[tuple(args)] = scaled_value
            return scaled_value

    def generate_gradient(self, *args: int):
        """
        generates a single "grid of noise", note: the number of points generated is based on the scale of the noise
        :param args: the coordinates for the gradient note: this is different from point coordinates (floor(point
            coordinates / scale) = gradient coordinates). THE COORDINATES MUST BE INTEGERS
        :return: an array of noise values representing the gradient
        """

    def generate_chunk(self, start: tuple, end: tuple):
        """
        generates a chunk of perlin noise (faster than generating each point 1 by 1)
        :param start: the starting corner of the noise to be generated
        :param end: the ending corner of the noise to be generated
        :return: an array of point values
        """

    def save(self):
        """
        saves the perlin noise object and all of its attributes
        :return: json data representing all of the class attributes as a string
        """

        return str(vars(self)).replace(', ', '\n').replace('{', '{\n').replace('}', '\n}')

    def load(self, data: str):
        """
        loads a previously saved PerlinNoise object with all of its previous attributes
        :param data: json data representing all of the class attributes as a string
        """

        # loads data
        self.saved_data = {}
        for line in data.split('\n'):
            pair = line.split(': ')
            if pair[0] == "'seed'":
                self.seed = float(pair[1])
            elif pair[0] == "'octaves'":
                self.octaves = []  # todo
            elif pair[0] == "'scale'":
                self.scale = float(pair[1])
            elif pair[0] == "'min'":
                self.min = float(pair[1])
            elif pair[0] == "'max'":
                self.max = float(pair[1])
            elif pair[0] == "'save_data'":
                self.save_data = bool(pair[1])
            elif pair[0] == "'saved_data'":
                pass
            elif len(pair) == 2:
                self.saved_data[eval(pair[0])] = float(pair[1])


class Octave(PerlinNoise):
    """
    a class for octaves of PerlinNoise objects with less parameter options and modified chunk generation to account for
    PerlinNoise scaling
    """

    def __init__(self, seed: float = None, scale: float = 255, minimum: float = -1, maximum: float = 1):
        """
        a modified init method of octaves limiting the number of options
        :param seed: the seed for the noise
        :param scale: the scale of the noise
        :param minimum: the minimum value of the noise
        :param maximum: the maximum value of the noise
        ** see PerlinNoise __init__ method for more info
        """

        super().__init__(seed, scale=scale, minimum=minimum, maximum=maximum)

    def __repr__(self):
        """
        returns the data of the octave instead of the location in memory
        :return: the json data of the octave represented as a string
        """

        return str({'seed': self.seed, 'scale': self.scale, 'min': self.min, 'max': self.max})
