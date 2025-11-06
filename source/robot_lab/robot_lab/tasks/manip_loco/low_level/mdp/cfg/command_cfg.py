from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
# from .. import pose_command , velocity_command
from ..pose_command import UniformPoseCommand, MLUniformPoseCommand
from ..velocity_command import UniformVelocityCommand, MLUniformVelocityCommand

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg as UniformPoseCommandBaseCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg as UniformVelocityCommandBaseCfg



@configclass
class UniformPoseCommandCfg(UniformPoseCommandBaseCfg):

    class_type: type = UniformPoseCommand

    is_Go2ARM: bool = False
    """ is_Go2ARM flag.
    check pose_command.py for more details
    """

    curriculum_coeff: int = MISSING
    """   
    The number of iterations for the pose command to linearly increase from range_init to range_final. 
    valid when is_Go2ARM = True
    check pose_command.py for more details
    """

    is_Go2ARM_Play: bool = False 
    """ is_Go2ARM_Play flag.
    check pose_command.py for more details
    """

    is_Go2ARM_Flat: bool = True #TODO
    """ 
    check pose_command.py for more details
    """

    ranges_init: UniformPoseCommandBaseCfg.Ranges = None
    """The initial range"""

    ranges_final: UniformPoseCommandBaseCfg.Ranges = None
    """The maximum range"""



@configclass
class UniformVelocityCommandCfg(UniformVelocityCommandBaseCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    is_Go2ARM: bool = False
    """ is_Go2ARM flag.
    check velocity_command.py for more details
    """

    curriculum_coeff: int = MISSING
    """   
    The number of iterations for the velocity command to linearly increase from range_init to range_final. 
    valid when is_Go2ARM = True
    check velocity_command.py for more details
    """    
    is_Go2ARM_Flat: bool = True #TODO
    """ 
    check velocity_command.py for more details
    """

    ranges_init: UniformVelocityCommandBaseCfg.Ranges = None
    """The initial range
    check velocity_command.py for more details"""
    
    ranges_final: UniformVelocityCommandBaseCfg.Ranges = None
    """The maximum range
    check velocity_command.py for more details"""


@configclass
class MLUniformPoseCommandCfg(UniformPoseCommandBaseCfg):

    class_type: type = MLUniformPoseCommand

    is_QuadrupedManipulator: bool = False
    """ is_QuadrupedManipulator flag.
    check pose_command.py for more details
    """

    curriculum_coeff: int = MISSING
    """   
    The number of iterations for the pose command to linearly increase from range_init to range_final. 
    valid when is_GQuadrupedManipulator = True
    check pose_command.py for more details
    """

    is_QuadrupedManipulator_Play: bool = False 
    """ is_QuadrupedManipulator_Play flag.
    check pose_command.py for more details
    """

    is_QuadrupedManipulator_Flat: bool = True #TODO
    """ 
    check pose_command.py for more details
    """

    ranges_init: UniformPoseCommandBaseCfg.Ranges = None
    """The initial range"""

    ranges_final: UniformPoseCommandBaseCfg.Ranges = None
    """The maximum range"""



@configclass
class MLUniformVelocityCommandCfg(UniformVelocityCommandBaseCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = MLUniformVelocityCommand

    is_QuadrupedManipulator: bool = False
    """ is_QuadrupedManipulator flag.
    check velocity_command.py for more details
    """

    curriculum_coeff: int = MISSING
    """   
    The number of iterations for the velocity command to linearly increase from range_init to range_final. 
    valid when is_QuadrupedManipulator = True
    check velocity_command.py for more details
    """    
    is_QuadrupedManipulator_Flat: bool = True #TODO
    """ 
    check velocity_command.py for more details
    """

    ranges_init: UniformVelocityCommandBaseCfg.Ranges = None
    """The initial range
    check velocity_command.py for more details"""
    
    ranges_final: UniformVelocityCommandBaseCfg.Ranges = None
    """The maximum range
    check velocity_command.py for more details"""