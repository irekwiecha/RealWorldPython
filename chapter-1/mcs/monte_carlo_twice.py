import sys
import random
import itertools as it
import numpy as np
import cv2 as cv

MAP_FILE = './images/cape_python.png'

SA1_CORNERS = (130, 265, 180, 315)  # (UL-X, UL-Y, LR-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305)   # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255)  # (UL-X, UL-Y, LR-X, LR-Y)


class Search:
    '''
    A Bayesian search and rescue mission simulation game with three search areas.
    '''

    def __init__(self, name):
        self.name = name
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        if self.img is None:
            print(f'Unable to load map file {MAP_FILE}', file=sys.stderr)
            sys.exit(1)

        # Creates attributes to store the actual location of a missing person.
        self.area_actual = 0

        # Local coordinates within the search area
        self.sailor_actual = [0, 0]

        # Creates a numpy array for each area, extracting ranges from the map.
        self.sa1 = self.img[SA1_CORNERS[1]: SA1_CORNERS[3],
                            SA1_CORNERS[0]: SA1_CORNERS[2]]
        self.sa2 = self.img[SA2_CORNERS[1]: SA2_CORNERS[3],
                            SA2_CORNERS[0]: SA2_CORNERS[2]]
        self.sa3 = self.img[SA3_CORNERS[1]: SA3_CORNERS[3],
                            SA3_CORNERS[0]: SA3_CORNERS[2]]

        # Specifies an initial estimate of the probability of finding a sailor for each area.
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

        # Initializes attributes that store search effectiveness.
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

    def sailor_final_location(self, num_search_areas):
        '''
        Returns the x and y coordinates of the real location of a missing person.
        '''
        # Finds the sailor's coordinates relative to the search area subarray.
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1], 1)
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0], 1)

        # Randomly searches for the area.
        area = int(random.triangular(1, num_search_areas + 1))

        # Converts the local coordinates of the search area to region map coordinates.
        if area == 1:
            x = self.sailor_actual[0] + SA1_CORNERS[0]
            y = self.sailor_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = self.sailor_actual[0] + SA2_CORNERS[0]
            y = self.sailor_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3:
            x = self.sailor_actual[0] + SA3_CORNERS[0]
            y = self.sailor_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3
        return x, y

    def calc_search_effectiveness(self):
        '''
        Designates a decimal value that represents the search effectiveness for each search area.
        '''
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def conduct_search(self, area_num, area_array, effectiveness_prob):
        '''
        Returns the search result and the list of coordinates searched.
        '''
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(it.product(local_x_range, local_y_range))
        random.shuffle(coords)
        coords = coords[:int(len(coords) * effectiveness_prob)]
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        if area_num == self.area_actual and loc_actual in coords:
            return f'Found in area {area_num}.', coords
        else:
            return 'Not found.', coords

    def revise_target_probs(self):
        '''
        Updates the probability for each area based on the effectiveness of the search.
        '''
        denom = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) + self.p3 * (1 - self.sep3)
        self.p1 = self.p1 * (1 - self.sep1) / denom
        self.p2 = self.p2 * (1 - self.sep2) / denom
        self.p3 = self.p3 * (1 - self.sep3) / denom

# Monte Carlo for twice 1 or 2 or 3


def main_twice(attempt):
    app = Search('Cape_Python')
    app.sailor_final_location(num_search_areas=3)
    search_num = 1
    while True:
        app.calc_search_effectiveness()
        to_choice = [app.p1, app.p2, app.p3]
        choice = str(to_choice.index(max(to_choice)) + 1)

        if choice == '0':
            sys.exit()

        elif choice == '1':
            result_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            result_2, coords_2 = app.conduct_search(1, app.sa1, app.sep1)
            app.sep1 = (len(set(coords_1 + coords_2)) / len(app.sa1) ** 2)
            app.sep2 = 0
            app.sep3 = 0

        elif choice == '2':
            result_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            result_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep1 = 0
            app.sep2 = (len(set(coords_1 + coords_2)) / len(app.sa2) ** 2)
            app.sep3 = 0

        elif choice == '3':
            result_1, coords_1 = app.conduct_search(3, app.sa3, app.sep3)
            result_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
            app.sep2 = 0
            app.sep3 = (len(set(coords_1 + coords_2)) / len(app.sa3) ** 2)

        elif choice == '4':
            result_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            result_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep3 = 0

        elif choice == '5':
            result_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            result_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep2 = 0

        elif choice == '6':
            result_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            result_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0

        # Uses Bayesian theory to update the probability.
        app.revise_target_probs()

        if result_1 != 'Not found.' or result_2 != 'Not found.':
            return search_num

        search_num += 1


if __name__ == '__main__':
    main_twice(1)
