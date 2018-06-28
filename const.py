from enum import Enum, unique


@unique
class Phase(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'
