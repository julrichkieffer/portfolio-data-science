# abstract base class work
from abc import ABC, abstractmethod


class UserAgent(ABC):

    @abstractmethod
    def get(self, url):
        pass
