from abc import abstractmethod
from typing import List, Optional
import numpy as np

class BaseIKSolver:
    @abstractmethod
    def step(self, pos: Optional[np.ndarray], quat: Optional[np.ndarray], repeat=1):
        pass

    @abstractmethod
    def get_current_qpos(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_current_qpos(self, qpos: np.ndarray):
        pass

    @abstractmethod
    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_ee_name(self) -> str:
        pass

    @abstractmethod
    def get_dof(self) -> int:
        pass

    @abstractmethod
    def get_timestep(self) -> float:
        pass

    @abstractmethod
    def get_joint_names(self) -> List[str]:
        pass

    @abstractmethod
    def is_use_gpu(self) -> bool:
        pass

