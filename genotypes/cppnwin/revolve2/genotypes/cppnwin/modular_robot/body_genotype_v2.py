import math
import multineat
import random
import operator
import sys
import pprint

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, Core, Module
from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1


def random_v1(
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
    multineat_params: multineat.Parameters,
    output_activation_func: multineat.ActivationFunction,
    num_initial_mutations: int,
    n_env_conditions: int,
    plastic_body: int,
) -> Genotype:
    if plastic_body == 0:
        return base_random_v1(
            innov_db,
            rng,
            multineat_params,
            output_activation_func,
            #3,  # bias(always 1), pos_x, pos_y
            2,  # pos_x, pos_y
            4,  # brick, activehinge, rot0, rot90
            num_initial_mutations,
        )
    else:
        return base_random_v1(
            innov_db,
            rng,
            multineat_params,
            output_activation_func,
            # 4,  # bias(always 1), pos_x, pos_y, inclined
            3,  #   pos_x, pos_y, inclined
            4,  # brick, activehinge, rot0, rot90
            num_initial_mutations,
        )


class Develop:

    def __init__(self, max_modules, substrate_radius, genotype, querying_seed, env_condition, n_env_conditions, plastic_body, bisymmetry):

        self.max_modules = max_modules
        self.quantity_modules = 0
        self.substrate_radius = substrate_radius
        self.genotype = genotype
        self.querying_seed = querying_seed
        self.env_condition = env_condition
        self.n_env_conditions = n_env_conditions
        self.plastic_body = plastic_body
        self.bisymmetry = bisymmetry
        self.development_seed = None
        self.random = None
        self.cppn = None
        # the queried substrate
        self.queried_substrate = {}
        self.phenotype_body = None
        self.parents_ids = []
        self.outputs_count = {
            'b_module': 0,
            'a_module': 0}

    def develop(self):

        self.random = random.Random(self.querying_seed)
        self.quantity_nodes = 0
        # the queried substrate
        self.queried_substrate = {}
        self.free_slots = {}
        self.outputs_count = {
            'b_module': 0,
            'a_module': 0}

        self.cppn = multineat.NeuralNetwork()
        self.genotype.genotype.BuildPhenotype(self.cppn)

        self.develop_body()
        self.phenotype_body.finalize()

        return self.phenotype_body, self.queried_substrate

    def develop_body(self):

        self.place_head()

        if self.bisymmetry == 0:
            self.attach_body()
        else:
            self.attach_body_bilateral()

        return self.phenotype_body

    def calculate_coordinates(self, parent, slot):
        # calculate the actual 2d direction and coordinates of new module using relative-to-parent position as reference
        dic = {Core.FRONT: 0,
               Core.LEFT: 1,
               Core.BACK: 2,
               Core.RIGHT: 3}

        inverse_dic = {0: Core.FRONT,
                       1: Core.LEFT,
                       2: Core.BACK,
                       3: Core.RIGHT}

        direction = dic[parent.turtle_direction] + dic[slot]
        if direction >= len(dic):
            direction = direction - len(dic)

        turtle_direction = inverse_dic[direction]
        if turtle_direction == Core.LEFT:
            coordinates = (parent.substrate_coordinates[0] - 1,
                           parent.substrate_coordinates[1])
        if turtle_direction == Core.RIGHT:
            coordinates = (parent.substrate_coordinates[0] + 1,
                           parent.substrate_coordinates[1])
        if turtle_direction == Core.FRONT:
            coordinates = (parent.substrate_coordinates[0],
                           parent.substrate_coordinates[1] + 1)
        if turtle_direction == Core.BACK:
            coordinates = (parent.substrate_coordinates[0],
                           parent.substrate_coordinates[1] - 1)

        return coordinates, turtle_direction

    def choose_free_slot(self):
        parent_module_coor = self.random.choice(list(self.free_slots.keys()))
        parent_module = self.queried_substrate[parent_module_coor]
        direction = self.random.choice(list(self.free_slots[parent_module_coor]))

        return parent_module_coor, parent_module, direction

    def attach_body(self):
        # size of substrate is (substrate_radius*2+1)^2
        parent_module_coor = (0, 0)

        self.free_slots[parent_module_coor] = [Core.LEFT,
                                               Core.FRONT,
                                               Core.RIGHT,
                                               Core.BACK]

        parent_module_coor, parent_module, direction = self.choose_free_slot()

        for q in range(0, self.max_modules):

            # calculates coordinates of potential new module
            potential_module_coord, turtle_direction = self.calculate_coordinates(parent_module, direction)

            radius = self.substrate_radius

            in_x_area = radius >= potential_module_coord[0] >= -radius
            in_y_area = radius >= potential_module_coord[1] >= -radius

            # substrate limit
            if not (in_x_area and in_y_area):

                self.free_slots[parent_module_coor].remove(direction)
                if len(self.free_slots[parent_module_coor]) == 0:
                    self.free_slots.pop(parent_module_coor)
                if len(self.free_slots)==0:
                    break
                else:
                    parent_module_coor, parent_module, direction = self.choose_free_slot()

            else:

                # queries potential new module given coordinates
                module_type, rotation = \
                    self.query_body_part(potential_module_coord[0], potential_module_coord[1])

                # if position in substrate is not already occupied
                if potential_module_coord in self.queried_substrate.keys():
                    self.free_slots[parent_module_coor].remove(direction)
                    if len(self.free_slots[parent_module_coor]) == 0:
                        self.free_slots.pop(parent_module_coor)
                    if len(self.free_slots) == 0:
                        break
                    else:
                        parent_module_coor, parent_module, direction = self.choose_free_slot()
                else:

                    new_module = self.new_module(module_type, rotation, parent_module)

                    new_module.substrate_coordinates = potential_module_coord

                    new_module.turtle_direction = turtle_direction
                    new_module.direction_from_parent = direction

                    # attaches module
                    parent_module.children[direction] = new_module
                    self.queried_substrate[potential_module_coord] = new_module

                    # joints branch out only to the front
                    if module_type is ActiveHinge:
                        directions = [ActiveHinge.ATTACHMENT]
                    else:
                        directions = [Core.LEFT,
                                       Core.FRONT,
                                       Core.RIGHT]

                    self.free_slots[parent_module_coor].remove(direction)
                    if len(self.free_slots[parent_module_coor]) == 0:
                        self.free_slots.pop(parent_module_coor)

                    # adds new slots to list of free slots
                    self.free_slots[potential_module_coord] = directions

                    # fins new free slot
                    parent_module_coor, parent_module, direction = self.choose_free_slot()

    def attach_body_bilateral(self):
        # size of substrate is (substrate_radius*2+1)^2
       # print('\n\n ----- ')
        parent_module_coor = (0, 0)

        self.free_slots[parent_module_coor] = [Core.LEFT,
                                               Core.FRONT,
                                               Core.RIGHT,
                                               Core.BACK]

        parent_module_coor, parent_module, direction = self.choose_free_slot()

        queries_plus_mirrored = 0
        for q in range(0, self.max_modules):

            if queries_plus_mirrored >= self.max_modules:
                break

            queries_plus_mirrored += 1

            # calculates coordinates of potential new module
            potential_module_coord, turtle_direction = self.calculate_coordinates(parent_module, direction)

            radius = self.substrate_radius

            # substrate limit
            in_x_area = 0 >= potential_module_coord[0] >= -radius
            in_y_area = radius >= potential_module_coord[1] >= -radius

            if not (in_x_area and in_y_area):

                self.free_slots[parent_module_coor].remove(direction)
                if len(self.free_slots[parent_module_coor]) == 0:
                    self.free_slots.pop(parent_module_coor)
                if len(self.free_slots)==0:
                    break
                else:
                    parent_module_coor, parent_module, direction = self.choose_free_slot()

            else:

                # queries potential new module given coordinates
                module_type, rotation = \
                    self.query_body_part(potential_module_coord[0], potential_module_coord[1])

                # if position in substrate is not already occupied
#                if potential_module_coord not in self.queried_substrate.keys():
                if potential_module_coord in self.queried_substrate.keys():
                    self.free_slots[parent_module_coor].remove(direction)
                    if len(self.free_slots[parent_module_coor]) == 0:
                        self.free_slots.pop(parent_module_coor)
                    if len(self.free_slots) == 0:
                        break
                    else:
                        parent_module_coor, parent_module, direction = self.choose_free_slot()
                else:

                    new_module = self.new_module(module_type, rotation, parent_module)

                    new_module.substrate_coordinates = potential_module_coord

                    new_module.turtle_direction = turtle_direction
                    new_module.direction_from_parent = direction

                    # attaches module
                    parent_module.children[direction] = new_module
                    self.queried_substrate[potential_module_coord] = new_module

                    # joints branch out only to the front
                    if module_type is ActiveHinge:
                        directions = [ActiveHinge.ATTACHMENT]
                    else:
                        directions = [Core.LEFT,
                                      Core.FRONT,
                                      Core.RIGHT]

                    self.free_slots[parent_module_coor].remove(direction)
                    if len(self.free_slots[parent_module_coor]) == 0:
                        self.free_slots.pop(parent_module_coor)

                    # adds new slots to list of free slots
                    self.free_slots[potential_module_coord] = directions

                    # mirroring
                    mirror_coord = (potential_module_coord[0] * -1, potential_module_coord[1])

                    if mirror_coord[0] > 0:

                        queries_plus_mirrored += 1

                        if type(parent_module) == Core:
                            mirrored_module = self.new_module(module_type, rotation, parent_module)
                            mirrored_module.substrate_coordinates = mirror_coord
                            direction = Core.RIGHT
                            mirrored_parent = parent_module

                            mirrored_module.direction_from_parent = direction
                            mirrored_parent.children[direction] = mirrored_module
                            self.queried_substrate[mirror_coord] = mirrored_module

                        else:

                            mirrored_parent = self.queried_substrate[parent_module.substrate_coordinates[0]*-1, parent_module.substrate_coordinates[1]]
                            mirrored_module = self.new_module(module_type, rotation, mirrored_parent)
                            mirrored_module.substrate_coordinates = mirror_coord

                            current_module = mirrored_parent
                            directions_backwards = []

                            # mapping segment backwards from mirrored parent until code
                            while type(current_module) != Core:
                                directions_backwards.append(current_module.direction_from_parent)
                                current_module = current_module._parent

                            directions_backwards.reverse()
                            previous_move = -1
                            turtle_orientation = Core.FRONT
                            current_forward = self.phenotype_body.core

                            for dir in directions_backwards:
                                if type(current_forward) == ActiveHinge:
                                    dir = ActiveHinge.ATTACHMENT

                                current_forward = current_forward.children[dir]

                                if (
                                        (dir == Core.FRONT and turtle_orientation == Core.FRONT) or
                                        (dir == Core.RIGHT and turtle_orientation == Core.LEFT) or
                                        (dir == Core.LEFT and turtle_orientation == Core.RIGHT) or
                                        (dir == Core.BACK and turtle_orientation == Core.BACK)):
                                    turtle_orientation = Core.FRONT
                                elif ((dir == Core.RIGHT and turtle_orientation == Core.FRONT) or
                                      (dir == Core.BACK and turtle_orientation == Core.LEFT) or
                                      (dir == Core.FRONT and turtle_orientation == Core.RIGHT) or
                                      (dir == Core.LEFT and turtle_orientation == Core.BACK)):
                                    turtle_orientation = Core.RIGHT
                                elif ((dir == Core.BACK and turtle_orientation == Core.FRONT) or
                                      (dir == Core.LEFT and turtle_orientation == Core.LEFT) or
                                      (dir == Core.RIGHT and turtle_orientation == Core.RIGHT) or
                                      (dir == Core.FRONT and turtle_orientation == Core.BACK)):
                                    turtle_orientation = Core.BACK
                                elif ((dir == Core.LEFT and turtle_orientation == Core.FRONT) or
                                      (dir == Core.FRONT and turtle_orientation == Core.LEFT) or
                                      (dir == Core.BACK and turtle_orientation == Core.RIGHT) or
                                      (dir == Core.RIGHT and turtle_orientation == Core.BACK)):
                                    turtle_orientation = Core.LEFT

                            # calculates orientation of new mirrored child relative to parent
                            p_x = mirrored_parent.substrate_coordinates[0]
                            c_x = mirrored_module.substrate_coordinates[0]
                            p_y = mirrored_parent.substrate_coordinates[1]
                            c_y = mirrored_module.substrate_coordinates[1]

                            if turtle_orientation == Core.RIGHT:
                                if p_y > c_y:
                                    turtle_orientation = Core.RIGHT
                                if p_x > c_x:
                                    turtle_orientation = Core.BACK
                                if p_x < c_x:
                                    turtle_orientation = Core.FRONT
                                if p_y < c_y:
                                    turtle_orientation = Core.LEFT

                            elif turtle_orientation == Core.FRONT:
                                if p_x < c_x:
                                    turtle_orientation = Core.RIGHT
                                if p_y > c_y:
                                    turtle_orientation = Core.BACK
                                if p_y < c_y:
                                    turtle_orientation = Core.FRONT
                                if p_x > c_x:
                                    turtle_orientation = Core.LEFT

                            elif turtle_orientation == Core.LEFT:
                                if p_y < c_y:
                                    turtle_orientation = Core.RIGHT
                                if p_x < c_x:
                                    turtle_orientation = Core.BACK
                                if p_x > c_x:
                                    turtle_orientation = Core.FRONT
                                if p_y > c_y:
                                    turtle_orientation = Core.LEFT

                            elif turtle_orientation == Core.BACK:
                                if p_x > c_x:
                                    turtle_orientation = Core.RIGHT
                                if p_y < c_y:
                                    turtle_orientation = Core.BACK
                                if p_y > c_y:
                                    turtle_orientation = Core.FRONT
                                if p_x < c_x:
                                    turtle_orientation = Core.LEFT

                            mirrored_module.direction_from_parent = turtle_orientation
                            if type(mirrored_parent) == ActiveHinge:
                                mirrored_parent.children[ActiveHinge.ATTACHMENT] = mirrored_module
                            else:
                                mirrored_parent.children[turtle_orientation] = mirrored_module
                            self.queried_substrate[mirror_coord] = mirrored_module

                    # fins new free slot
                    parent_module_coor, parent_module, direction = self.choose_free_slot()

    def place_head(self):

        module_type = Core
        self.phenotype_body = Body()
        self.phenotype_body.core.turtle_direction = Core.FRONT
        orientation = 0
        self.phenotype_body.core._id = self.quantity_modules
        self.phenotype_body.core._rotation = orientation * (math.pi / 2.0)
        self.phenotype_body.core._orientation = 0
        self.phenotype_body.core.rgb = self.get_color(module_type, orientation)
        self.phenotype_body.core.substrate_coordinates = (0, 0)
        self.queried_substrate[(0, 0)] = self.phenotype_body.core

    def new_module(self, module_type, orientation, parent_module):

        # calculates _absolute_rotation
        absolute_rotation = 0
        if module_type == ActiveHinge and orientation == 1:
            if type(parent_module) == ActiveHinge and parent_module._absolute_rotation == 1:
                absolute_rotation = 0
            else:
                absolute_rotation = 1
        else:
            if type(parent_module) == ActiveHinge and parent_module._absolute_rotation == 1:
                absolute_rotation = 1

        # makes sure it wont rotate bricks, so to prevent 3d shapes
        if module_type == Brick and type(parent_module) == ActiveHinge and parent_module._absolute_rotation == 1:
            # inverts it no absolute rotation
            orientation = 1

        module = module_type(orientation * (math.pi / 2.0))
        self.quantity_modules += 1
        module._id = str(self.quantity_modules)
        module._absolute_rotation = absolute_rotation
        module.rgb = self.get_color(module_type, orientation)
        module._parent = parent_module
        return module
    
    def query_body_part(self, x_dest, y_dest):

        # # Applies regulation according to environmental conditions.
        if self.plastic_body == 0:
            # self.cppn.Input(  [1.0, x_dest, y_dest]  )  # 1.0 is the bias input
            self.cppn.Input([x_dest, y_dest])
        else:

            staticfriction, dynamicfriction, yrotationdegrees, platform, toxic  = \
                float(self.env_condition[0]), \
                float(self.env_condition[1]), \
                float(self.env_condition[2]), \
                float(self.env_condition[3]), \
                float(self.env_condition[4])

            # TODO: make conditions-checking dynamic

            # if inclined
            # if yrotationdegrees > 0:
            #     inclined = -1
            # else:
            #     inclined = 1

            # obsolete name: toxic here means just a change in task
            if toxic > 0:
                toxicenv = 1
            else:
                toxicenv = -1

            self.cppn.Input(
                #[x_dest, y_dest, inclined]
                [x_dest, y_dest, toxicenv]
                #   [1.0, x_dest, y_dest, inclined]  # 1.0 is the bias input
            )

        self.cppn.ActivateAllLayers()
        outputs = self.cppn.Output()

        # get module type from output probabilities
        type_probs = [outputs[0], outputs[1]]
        types = [Brick, ActiveHinge]
        module_type = types[type_probs.index(max(type_probs))]

        # get rotation from output probabilities
        if module_type is ActiveHinge:
            rotation_probs = [outputs[2], outputs[3]]
            rotation = rotation_probs.index(max(rotation_probs))
        else:
            rotation = 0

        return module_type, rotation

    def get_color(self, module_type, rotation):
        rgb = []
        if module_type == Brick:
            rgb = [0, 0, 1]
        if module_type == ActiveHinge:
            if rotation == 0:
                rgb = [1, 0.08, 0.58]
            else:
                rgb = [0.7, 0, 0]
        if module_type == Core:
            rgb = [1, 1, 0]
        return rgb
