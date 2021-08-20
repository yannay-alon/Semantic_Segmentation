import abc


class BaseModel:

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError("The fit method must be implemented")

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError("The predict method must be implemented")
