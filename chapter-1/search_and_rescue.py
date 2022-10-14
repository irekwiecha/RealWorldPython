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
    A Bayesian search and rescue mission simulation game with 3 search areas.
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

    def draw_map(self, last_known):
        '''
        Displays a map of the region with scale, last known location and search areas.
        '''
        # Draws a scale bar.
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)
        cv.putText(self.img, '0', (8, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv.putText(self.img, '50 sea miles', (71, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        # Draws and numbers the search areas.
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '1', (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]), (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '2', (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]), (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3', (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        # Marks the last known location of the missing person on the map.
        cv.putText(self.img, '+', last_known, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '+ = last known location', (240, 355), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '* = actual location', (242, 370), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        cv.imshow('Areas to be searched', self.img)
        cv.moveWindow('Areas to be searched', 750, 10)
        cv.waitKey(500)

    def sailor_final_location(self, num_search_areas):
        '''
        Returns the x and y coordinates of the real location of a missing person.
        '''
        # Finds the sailor's coordinates relative to the search area subarray.
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1])
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0])

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


def draw_menu(search_num):
    '''
    Prints a menu with a selection of the area to be searched.
    '''
    print(f'\nApproach No. {search_num}')
    print(
        '''
        Select the next areas to search:
        0 - Exit the program
        1 - Search area 1 twice
        2 - Search area 2 twice
        3 - Search area 3 twice
        4 - Search areas 1 & 2
        5 - Search areas 1 & 3
        6 - Search areas 2 & 3
        7 - Start all over again
        '''
    )


def main():
    app = Search('Cape_Python')
    app.draw_map(last_known=(160, 290))
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)
    print('-' * 65)
    print('\nInitial probability estimate (P):')
    print(f'P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}')
    search_num = 1

    while True:
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = input('Choose an option: ')

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

        elif choice == '7':
            main()

        else:
            print('\nIt is not a valid choice.', file=sys.stderr)
            continue

        # Uses Bayesian theory to update the probability.
        app.revise_target_probs()

        print(f'\nApproach No. {search_num} - result 1: {result_1}', file=sys.stderr)
        print(f'Approach No. {search_num} - result 2: {result_2}', file=sys.stderr)
        print(f'Search effectiveness (E) for approach nr {search_num}')
        print(f'E1 = {app.sep1:.3f}, E2 = {app.sep2:.3f}, E3 = {app.sep3:.3f}')

        # Prints the updated probability value if the sailor is not found.
        # Otherwise it shows the position.
        if result_1 == 'Not found.' and result_2 == 'Not found.':
            print(f'\nNew probability (P) estimate for approach nr {search_num + 1}')
            print(f'P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}')
        else:
            cv.circle(app.img, (sailor_x, sailor_y), 3, (255, 0, 0), -1)
            cv.imshow('Areas to be searched', app.img)
            cv.waitKey(0)
            main()

        search_num += 1


if __name__ == '__main__':
    main()
