import math
from revolve2.core.modular_robot import ModularRobot, ActiveHinge, Body, Brick

def make_body_spider():
    body_spider = Body()
    body_spider.core._id = 1

    ###
    body_spider.core.front = ActiveHinge(math.pi / 2.0)
    body_spider.core.front._id = 2
    body_spider.core.front._absolute_rotation = 90

    body_spider.core.front.attachment = Brick(0.0)
    body_spider.core.front.attachment._id = 3
    body_spider.core.front.attachment._absolute_rotation = 0

    body_spider.core.front.attachment.front = ActiveHinge(math.pi / 2.0)
    body_spider.core.front.attachment.front._id = 4
    body_spider.core.front.attachment.front._absolute_rotation = 0

    body_spider.core.front.attachment.front.attachment = Brick(0.0)
    body_spider.core.front.attachment.front.attachment._id = 5
    body_spider.core.front.attachment.front.attachment._absolute_rotation = 0

    ###
    body_spider.core.right = ActiveHinge(math.pi / 2.0)
    body_spider.core.right._id = 6
    body_spider.core.right._absolute_rotation = 90

    body_spider.core.right.attachment = Brick(0.0)
    body_spider.core.right.attachment._id = 7
    body_spider.core.right.attachment._absolute_rotation = 0

    body_spider.core.right.attachment.front = ActiveHinge(math.pi / 2.0)
    body_spider.core.right.attachment.front._id = 8
    body_spider.core.right.attachment.front._absolute_rotation = 0

    body_spider.core.right.attachment.front.attachment = Brick(0.0)
    body_spider.core.right.attachment.front.attachment._id = 9
    body_spider.core.right.attachment.front.attachment._absolute_rotation = 0

    ###
    body_spider.core.back = ActiveHinge(math.pi / 2.0)
    body_spider.core.back._id = 10
    body_spider.core.back._absolute_rotation = 90

    body_spider.core.back.attachment = Brick(0.0)
    body_spider.core.back.attachment._id = 11
    body_spider.core.back.attachment._absolute_rotation = 0

    body_spider.core.back.attachment.front = ActiveHinge(math.pi / 2.0)
    body_spider.core.back.attachment.front._id = 12
    body_spider.core.back.attachment.front._absolute_rotation = 0

    body_spider.core.back.attachment.front.attachment = Brick(0.0)
    body_spider.core.back.attachment.front.attachment._id = 13
    body_spider.core.back.attachment.front.attachment._absolute_rotation = 0

    ###
    body_spider.core.left = ActiveHinge(math.pi / 2.0)
    body_spider.core.left._id = 14
    body_spider.core.left._absolute_rotation = 90

    body_spider.core.left.attachment = Brick(0.0)
    body_spider.core.left.attachment._id = 15
    body_spider.core.left.attachment._absolute_rotation = 0

    body_spider.core.left.attachment.front = ActiveHinge(math.pi / 2.0)
    body_spider.core.left.attachment.front._id = 16
    body_spider.core.left.attachment.front._absolute_rotation = 0

    body_spider.core.left.attachment.front.attachment = Brick(0.0)
    body_spider.core.left.attachment.front.attachment._id = 17
    body_spider.core.left.attachment.front.attachment._absolute_rotation = 0

    body_spider.finalize()

    return body_spider